"""
tests/test_leakage.py
---------------------
Tests for verify/leakage.py:
  - Clean redacted text produces downstream_safe=True (0 leaks).
  - Placeholder tokens are not falsely flagged as PII.
  - Injected leftover PII produces downstream_safe=False and lists leaks.
  - Blocked chunks are handled safely without errors.
"""
from pathlib import Path
import pytest
import yaml

from extract.models import Chunk
from verify.leakage import check_leakage, _strip_placeholders

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


@pytest.fixture
def config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_strip_placeholders():
    text = "Contact [PERSON_1] at [EMAIL_1] or call [PHONE_1]."
    stripped = _strip_placeholders(text)
    assert "[PERSON_1]" not in stripped
    assert "[EMAIL_1]" not in stripped
    assert "[PHONE_1]" not in stripped
    assert "Contact" in stripped and "call" in stripped


def test_clean_redacted_text_is_safe(config):
    """A chunk containing only non-PII and placeholder tokens should pass cleanly."""
    chunks = [
        Chunk(
            file="test_doc.txt",
            location="page:1",
            text="The report was approved by [PERSON_1]. You can reach out at [EMAIL_1].",
            method="native",
            ocr_confidence=None,
            order_index=0,
        )
    ]
    result = check_leakage(chunks, config)
    assert result.downstream_safe is True
    assert len(result.leaks) == 0
    assert "downstream_safe=YES" in result.summary()


def test_leaked_ssn_detected(config):
    """A chunk containing unredacted SSN should be flagged as a leak."""
    chunks = [
        Chunk(
            file="test_doc.txt",
            location="page:1",
            text="The participant SSN is 219-45-6789 and was reviewed by [PERSON_1].",
            method="native",
            ocr_confidence=None,
            order_index=0,
        )
    ]
    result = check_leakage(chunks, config)
    assert result.downstream_safe is False
    assert len(result.leaks) > 0
    leak_types = [e.entity_type for e in result.leaks]
    assert "SSN" in leak_types


def test_leaked_email_detected(config):
    """A chunk containing unredacted email should be flagged as a leak."""
    chunks = [
        Chunk(
            file="test_doc.txt",
            location="page:2",
            text="Send replies directly to secret.agent@cadence-corp.com promptly.",
            method="native",
            ocr_confidence=None,
            order_index=1,
        )
    ]
    result = check_leakage(chunks, config)
    assert result.downstream_safe is False
    leak_types = [e.entity_type for e in result.leaks]
    assert "EMAIL" in leak_types


def test_blocked_chunk_handled_safely(config):
    """Blocked chunks should be skipped during leakage check."""
    chunks = [
        Chunk(
            file="test_doc.pdf",
            location="page:3",
            text="",
            method="blocked",
            ocr_confidence=None,
            order_index=2,
            blocked_reason="OCR confidence below threshold",
        )
    ]
    result = check_leakage(chunks, config)
    assert result.downstream_safe is True
    assert len(result.leaks) == 0


# ---------------------------------------------------------------------------
# New leakage tests (P2 fixes)
# ---------------------------------------------------------------------------

def test_strip_placeholders_uses_block_symbol():
    """Placeholder substitution must use '██' not ',' to avoid false regex matches."""
    text = "Call [PHONE_1] for details."
    stripped = _strip_placeholders(text)
    # Must not contain commas that could form phone/date false positives
    assert "," not in stripped
    # Must contain the block symbol
    assert "██" in stripped


def test_placeholder_substitution_does_not_produce_ip_false_positive(config):
    """'██' separator between numbers must not match IP address regex."""
    from detect.rules import run_rules
    # Simulate a text where placeholders are replaced
    text = "Values: ██ 55 ██ 3 ██ 22"
    hits = run_rules(text, config)
    ips = [h for h in hits if h.entity_type == "IP_ADDRESS"]
    assert len(ips) == 0, "Block symbol separator must not create IP address false positives"


def test_simplename_space_scan_chunk_works(config):
    """The SimpleNamespace scan chunk must behave identically to a real Chunk."""
    from types import SimpleNamespace
    from detect.merge import detect_chunk

    # Build a SimpleNamespace the same way leakage.py does
    scan_chunk = SimpleNamespace(
        file="test.txt",
        location="page:1",
        text="Contact alice@example.com for help.",
        method="native",
        ocr_confidence=None,
        order_index=0,
        blocked_reason=None,
        is_blocked=lambda: False,
    )
    entities = detect_chunk(scan_chunk, config)
    types = {e.entity_type for e in entities}
    assert "EMAIL" in types, "SimpleNamespace scan chunk must work with detect_chunk"


def test_all_blocked_chunks_return_zero_leaks(config):
    """A document with all chunks blocked must produce safe=True with no errors."""
    chunks = [
        Chunk(
            file="blocked_doc.pdf",
            location=f"page:{i}",
            text="",
            method="blocked",
            ocr_confidence=5.0,
            order_index=i,
            blocked_reason="OCR too low",
        )
        for i in range(1, 4)
    ]
    result = check_leakage(chunks, config)
    assert result.downstream_safe is True
    assert len(result.leaks) == 0


def test_partially_redacted_ssn_flagged(config):
    """A chunk with a partially-redacted SSN must be flagged as unsafe."""
    chunk = Chunk(
        file="ssn_leak.txt",
        location="page:1",
        text="Participant record: SSN was partially hidden 219-45-[SSN_1] in error.",
        method="native",
        ocr_confidence=None,
        order_index=0,
    )
    result = check_leakage([chunk], config, known_values=["219-45-6789"])
    assert result.downstream_safe is False
    assert len(result.leaks) > 0
    types = [e.entity_type for e in result.leaks]
    assert "SSN" in types


def test_partially_redacted_ssn_without_known_values(config):
    """Partially redacted SSN like 219-45-XXXX is caught by independent regex."""
    chunk = Chunk(
        file="ssn_leak2.txt",
        location="page:1",
        text="Participant SSN was 219-45-XXXX on file.",
        method="native",
        ocr_confidence=None,
        order_index=0,
    )
    result = check_leakage([chunk], config)
    assert result.downstream_safe is False
    assert len(result.leaks) > 0
    assert any(e.entity_type in ("SSN", "GOVERNMENT_ID") for e in result.leaks)


def test_known_value_propagation(config):
    """Known values from mapping file are flagged if they appear in redacted text (even lowercase)."""
    chunk = Chunk(
        file="name_leak.txt",
        location="page:1",
        text="Author was noted as emily watson in the appendix.",
        method="native",
        ocr_confidence=None,
        order_index=0,
    )
    # Passed with different casing
    result = check_leakage([chunk], config, known_values=["Emily Watson"])
    assert result.downstream_safe is False
    assert len(result.leaks) > 0
    # Residue must contain masked version
    assert any("E**********n" in r or "e**********n" in r for r in result.residue)


def test_unverified_residue_reporting(config):
    """Initials or long digit runs outside placeholders are captured in unverified residue."""
    chunk = Chunk(
        file="residue_doc.txt",
        location="page:1",
        text="The system batch 887766554433 was processed with PQ Lindqvist in sector 4.",
        method="native",
        ocr_confidence=None,
        order_index=0,
    )
    result = check_leakage([chunk], config)
    assert len(result.unverified_residue) > 0
    assert len(result.residue) > 0
    # Masked values only: no raw numbers or full name
    assert "9988776655" not in result.residue


def test_residue_reporting_masks_all_values(config):
    """All reported residues must be masked, zero raw PII in residue list."""
    chunk = Chunk(
        file="raw_check.txt",
        location="page:1",
        text="Leak here: secret.agent@cadence-corp.com and 219-45-6789.",
        method="native",
        ocr_confidence=None,
        order_index=0,
    )
    result = check_leakage([chunk], config)
    assert result.downstream_safe is False
    for res_str in result.residue:
        assert "secret.agent@cadence-corp.com" not in res_str
        assert "219-45-6789" not in res_str
        assert "*" in res_str


def test_zero_raw_pii_in_pipeline_logs_and_reports(config, tmp_path, caplog):
    """
    Item 4 security guarantee:
    Scans every log line, report, and exported mapping file produced by a full run
    for raw values listed in ground truth. Raw values must never appear in logs or reports.
    """
    import json
    import logging
    from extract.router import route
    from detect.merge import detect_chunk
    from detect.propagate import propagate_name_aliases
    from redact.redactor import redact_chunks
    from redact.mapping import PlaceholderMapping
    from verify.leakage import check_leakage
    from report.report import build_file_summary, generate_report

    # Load ground truth raw values
    gt_file = ROOT / "synthetic" / "ground_truth" / "training_guide.json"
    assert gt_file.exists(), "Ground truth file missing"
    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Collect raw values to check (skip very short or trivial words)
    raw_secrets = set()
    for ent in gt_data.get("entities", []):
        raw = ent.get("text", "").strip()
        if len(raw) >= 4:
            raw_secrets.add(raw)

    assert len(raw_secrets) > 0, "No raw ground truth secrets found"

    # Capture logs during the run
    with caplog.at_level(logging.DEBUG):
        doc_path = ROOT / "synthetic" / "training_guide.docx"
        chunks = route(str(doc_path), config)

        file_entities = []
        for chunk in chunks:
            if not chunk.is_blocked():
                file_entities.extend(detect_chunk(chunk, config))

        # Propagate aliases
        person_ents = [e for e in file_entities if e.entity_type == "PERSON"]
        chunk_texts = [c.text for c in chunks if not c.is_blocked()]
        chunk_locs = [c.location for c in chunks if not c.is_blocked()]
        alias_ents = propagate_name_aliases(
            chunk_texts, person_ents, file_entities, config,
            file=str(doc_path), locations=chunk_locs,
        )
        file_entities.extend(alias_ents)

        # Redact with encrypted mapping
        out_dir = tmp_path / "out_audit"
        mapping = PlaceholderMapping(source_file=str(doc_path), output_dir=str(out_dir))
        redacted = redact_chunks(chunks, file_entities, mapping, config)
        mapping.save()

        # Leak check
        leak_res = check_leakage(redacted, config, known_values=mapping)

        # Build summary and generate report
        summary = build_file_summary(str(doc_path), chunks, file_entities, leak_res, structure_score=0.95)
        rep_paths = generate_report([summary], file_entities, str(out_dir))

    # 1. Check all log records: ZERO unmasked raw PII allowed in logs
    all_log_text = " ".join([rec.getMessage() for rec in caplog.records])
    for secret in raw_secrets:
        assert secret not in all_log_text, f"LEAK IN LOGS: Raw ground truth value '{secret}' found in pipeline logs!"

    # 2. Check generated report files (CSV and JSON): ZERO raw PII allowed in reports
    with open(rep_paths["csv"], "r", encoding="utf-8") as f:
        csv_text = f.read()
    with open(rep_paths["json"], "r", encoding="utf-8") as f:
        json_text = f.read()

    for secret in raw_secrets:
        assert secret not in csv_text, f"LEAK IN CSV REPORT: Raw value '{secret}' found in CSV report!"
        assert secret not in json_text, f"LEAK IN JSON REPORT: Raw value '{secret}' found in JSON report!"

    # 3. Check mapping file at rest on disk: must be encrypted (ciphertext)
    assert mapping.mapping_path.exists()
    raw_mapping_bytes = mapping.mapping_path.read_bytes()
    for secret in raw_secrets:
        assert secret.encode("utf-8") not in raw_mapping_bytes, \
            f"LEAK IN MAPPING AT REST: Raw value '{secret}' found unencrypted in mapping file!"

    # 4. Clean up / purge
    mapping.purge()
    assert not mapping.mapping_path.exists()



