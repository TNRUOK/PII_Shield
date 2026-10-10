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

