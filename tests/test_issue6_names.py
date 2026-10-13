"""
tests/test_issue6_names.py
--------------------------
Unit test suite verifying Issue 6:
- Anchor rule: full name directly before EMP/DIR/email/phone is detected as PERSON
- Email-derived surnames: surname and initial+surname propagate document-wide
- Alias propagation: surnames propagate document-wide
- Non-person terms: Group, Committee, Office, etc. are untouched
"""
import pytest
import yaml
from pathlib import Path

from extract.models import Chunk
from detect.context import run_context
from detect.propagate import propagate_name_aliases
from detect.merge import detect_chunk, Entity


@pytest.fixture
def config():
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_anchor_rule_detects_names_before_credentials(config):
    text = (
        "Designated Leader | Official Credentials | Direct Contact\n"
        "M. T. Achebe | m.achebe@cadencecorp.net (EMP-88210) | +44 20 7555 0171\n"
        "T. Lindqvist | t.lindqvist@cadencecorp.net (EMP-99341) | +46 8 5555 0121\n"
        "P. S. Raghavan | p.raghavan@cadencecorp.net (EMP-77124) | +91 80 4555 0182\n"
        "Kavita Reddy | k.reddy@cadencecorp.net (EMP-66219) | +91 22 4555 0117\n"
        "Elena Rostova (DIR-4421) | elena.rostova@example.org"
    )
    new_hits, _ = run_context("table:1", text, [], config)
    detected_names = [h.text for h in new_hits if h.entity_type == "PERSON"]
    assert "M. T. Achebe" in detected_names
    assert "T. Lindqvist" in detected_names
    assert "P. S. Raghavan" in detected_names
    assert "Kavita Reddy" in detected_names
    assert "Elena Rostova" in detected_names


def test_anchor_rule_detects_names_after_role_title(config):
    text = "Primary Signatory: Marcus Vance approved the executive resolution."
    new_hits, _ = run_context("body:1", text, [], config)
    detected_names = [h.text for h in new_hits if h.entity_type == "PERSON"]
    assert "Marcus Vance" in detected_names


def test_email_derived_surnames_propagate(config):
    # Simulated document chunks
    c1_text = "Primary contact is m.achebe@cadencecorp.net for London operations."
    c2_text = "Authority is delegated to Achebe for UK incident sign-offs. Furthermore, M. Achebe approved the change."
    chunks_text = [c1_text, c2_text]
    locations = ["body:1", "body:2"]

    chunk1 = Chunk(file="test.docx", location="body:1", text=c1_text, method="native", ocr_confidence=None, order_index=0)
    ents_c1 = detect_chunk(chunk1, config)

    # propagate
    alias_ents = propagate_name_aliases(
        chunks_text, [], ents_c1, config, file="test.docx", locations=locations
    )
    alias_texts = [e.raw_value for e in alias_ents]
    assert "Achebe" in alias_texts
    assert "M. Achebe" in alias_texts


def test_surname_propagation_case_sensitive(config):
    # Case-sensitive propagation should NOT match lowercase words
    c1_text = "Primary contact is T. Lindqvist at Stockholm."
    c2_text = "Reported by Lindqvist yesterday. The word lindqvist in lowercase should not be treated as a proper name."
    chunks_text = [c1_text, c2_text]
    locations = ["body:1", "body:2"]

    chunk1 = Chunk(file="test.docx", location="body:1", text=c1_text, method="native", ocr_confidence=None, order_index=0)
    ents_c1 = [
        Entity(
            entity_type="PERSON",
            sensitivity="Medium",
            confidence=0.9,
            methods=["rule"],
            file="test.docx",
            location="body:1",
            start=19,
            end=31,
            raw_value="T. Lindqvist",
            masked_value="T. L********",
            placeholder="[PERSON_1]",
            chunk_index=0,
        )
    ]
    alias_ents = propagate_name_aliases(
        chunks_text, ents_c1, ents_c1, config, file="test.docx", locations=locations
    )
    matched_values = [e.raw_value for e in alias_ents]
    assert "Lindqvist" in matched_values
    assert "lindqvist" not in matched_values


def test_non_person_terms_never_propagate(config):
    text = (
        "Board Risk Committee and Data Privacy Office reviewed Security Group settings. "
        "Also USER GROUPS were updated by Data Quality Group."
    )
    chunk = Chunk(file="test.docx", location="body:1", text=text, method="native", ocr_confidence=None, order_index=0)
    ents = detect_chunk(chunk, config)
    alias_ents = propagate_name_aliases(
        [text], ents, ents, config, file="test.docx", locations=["body:1"]
    )
    all_ents = ents + alias_ents
    person_texts = [e.raw_value for e in all_ents if e.entity_type == "PERSON"]
    for forbidden in ["Security Group", "Group", "Committee", "Board Risk Committee", "Data Privacy Office", "USER GROUPS"]:
        assert forbidden not in person_texts

