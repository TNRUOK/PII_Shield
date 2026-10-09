"""
tests/test_detection.py
------------------------
Unit tests for detection layers: rules, NER, context, and the full detect_chunk pipeline.
Uses synthetic text samples — no file I/O for most tests.
"""
import pytest
from detect.rules import run_rules, RawHit
from detect.context import run_context
from detect.merge import detect_chunk, _mask
from extract.models import Chunk


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _chunk(text: str, location: str = "body", file: str = "test.docx") -> Chunk:
    return Chunk(
        file=file, location=location, text=text,
        method="native", ocr_confidence=None, order_index=0,
    )


# ---------------------------------------------------------------------------
# Rule layer tests
# ---------------------------------------------------------------------------

class TestEmailRule:
    def test_simple_email(self, config):
        hits = run_rules("Contact jane.doe@example.com for support.", config)
        emails = [h for h in hits if h.entity_type == "EMAIL"]
        assert len(emails) == 1
        assert "jane.doe@example.com" in emails[0].text

    def test_no_false_positive_domain_only(self, config):
        hits = run_rules("Visit example.com for help.", config)
        emails = [h for h in hits if h.entity_type == "EMAIL"]
        assert len(emails) == 0


class TestPhoneRule:
    def test_us_phone(self, config):
        hits = run_rules("Call (555) 123-4567 now.", config)
        phones = [h for h in hits if h.entity_type == "PHONE"]
        assert len(phones) >= 1

    def test_india_phone(self, config):
        hits = run_rules("Mobile: +91 9876543210", config)
        phones = [h for h in hits if h.entity_type == "PHONE"]
        assert len(phones) >= 1


class TestSSNRule:
    def test_valid_ssn(self, config):
        hits = run_rules("SSN: 123-45-6789", config)
        ssns = [h for h in hits if h.entity_type == "SSN"]
        assert len(ssns) == 1

    def test_invalid_area_000_not_flagged(self, config):
        hits = run_rules("ID: 000-45-6789", config)
        ssns = [h for h in hits if h.entity_type == "SSN"]
        assert len(ssns) == 0, "SSN with area 000 must be rejected"

    def test_invalid_area_666_not_flagged(self, config):
        hits = run_rules("Ref: 666-45-6789", config)
        ssns = [h for h in hits if h.entity_type == "SSN"]
        assert len(ssns) == 0


class TestCreditCardRule:
    def test_valid_visa_flagged(self, config):
        hits = run_rules("Card: 4111 1111 1111 1111", config)
        cards = [h for h in hits if h.entity_type == "CREDIT_CARD"]
        assert len(cards) == 1

    def test_invalid_luhn_not_flagged(self, config):
        """4111 1111 1111 1112 fails Luhn — must NOT be flagged."""
        hits = run_rules("Test card: 4111 1111 1111 1112", config)
        cards = [h for h in hits if h.entity_type == "CREDIT_CARD"]
        assert len(cards) == 0, "Invalid Luhn card must be hard-rejected"


class TestEmployeeIDRule:
    def test_emp_prefix_style(self, config):
        hits = run_rules("EMP-12345 is the employee identifier.", config)
        emp = [h for h in hits if h.entity_type == "EMPLOYEE_ID"]
        assert len(emp) >= 1

    def test_labelled_form(self, config):
        hits = run_rules("Employee ID: EMP-98765", config)
        emp = [h for h in hits if h.entity_type == "EMPLOYEE_ID"]
        assert len(emp) >= 1


class TestIPAddressRule:
    def test_valid_ip(self, config):
        hits = run_rules("Workstation IP: 10.0.1.55", config)
        ips = [h for h in hits if h.entity_type == "IP_ADDRESS"]
        assert len(ips) == 1

    def test_invalid_octet_not_flagged(self, config):
        hits = run_rules("Bad IP: 999.0.0.1", config)
        ips = [h for h in hits if h.entity_type == "IP_ADDRESS"]
        assert len(ips) == 0


# ---------------------------------------------------------------------------
# Context layer tests
# ---------------------------------------------------------------------------

class TestContextLayer:
    def test_table_column_header_boost(self, config):
        """Hits in a 'Name' column should get confidence boost."""
        location = "table:1/r2[headers: Name | Email | Phone]"
        text = "Alice Johnson | alice@example.com | 555-123-4567"
        hits = run_rules(text, config)
        original_confs = {h.entity_type: h.confidence for h in hits}
        _, boosted = run_context(location, text, hits, config)
        # At least one hit in a PII column should have been boosted
        boosted_confs = {h.entity_type: h.confidence for h in boosted}
        # EMAIL and PHONE are in PII columns → should be boosted or same
        for etype in ("EMAIL", "PHONE"):
            if etype in original_confs:
                assert boosted_confs.get(etype, 0) >= original_confs[etype]

    def test_labelled_name_pattern(self, config):
        location = "body"
        text = "Prepared by: John Smith"
        hits = run_rules(text, config)
        new_hits, _ = run_context(location, text, hits, config)
        persons = [h for h in new_hits if h.entity_type == "PERSON"]
        assert len(persons) >= 1, "Labelled name 'Prepared by: John Smith' must be detected"

    def test_approved_by_pattern(self, config):
        location = "body"
        text = "Approved by: Mary Johnson, Head of Risk"
        _, _ = run_context(location, text, [], config)
        new_hits, _ = run_context(location, text, [], config)
        persons = [h for h in new_hits if h.entity_type == "PERSON"]
        assert any("Mary" in h.text for h in persons)


# ---------------------------------------------------------------------------
# detect_chunk (full pipeline)
# ---------------------------------------------------------------------------

class TestDetectChunk:
    def test_email_detected(self, config):
        chunk = _chunk("Send report to bob@acme.com by Friday.")
        ents = detect_chunk(chunk, config)
        types = {e.entity_type for e in ents}
        assert "EMAIL" in types

    def test_blocked_chunk_returns_empty(self, config):
        chunk = Chunk(
            file="test.pdf", location="page:1", text="",
            method="blocked", ocr_confidence=10.0, order_index=0,
            blocked_reason="Low confidence",
        )
        ents = detect_chunk(chunk, config)
        assert ents == []

    def test_no_pii_returns_empty(self, config):
        chunk = _chunk("The quick brown fox jumps over the lazy dog.")
        ents = detect_chunk(chunk, config)
        # Should have zero or very few hits (no deliberate PII)
        pii_types = {e.entity_type for e in ents}
        assert "SSN" not in pii_types
        assert "CREDIT_CARD" not in pii_types

    def test_entity_has_masked_value(self, config):
        chunk = _chunk("Email: test.person@example.com")
        ents = detect_chunk(chunk, config)
        emails = [e for e in ents if e.entity_type == "EMAIL"]
        if emails:
            assert "*" in emails[0].masked_value
            assert emails[0].raw_value != ""


class TestMaskFunction:
    def test_long_string(self):
        assert _mask("Jane Smith") == "J********h"

    def test_two_chars(self):
        assert _mask("AB") == "**"

    def test_one_char(self):
        assert _mask("X") == "*"

    def test_email_masking(self):
        masked = _mask("bob@example.com")
        assert masked[0] == "b"
        assert masked[-1] == "m"
        assert "*" in masked
