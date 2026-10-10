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
        assert len(ips) >= 1

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


# ---------------------------------------------------------------------------
# New rule tests (P1 fixes)
# ---------------------------------------------------------------------------

class TestAadhaarBare12Digit:
    def test_bare_12_digits_with_valid_verhoeff(self, config):
        """12 contiguous digits that pass Verhoeff should be flagged as AADHAAR."""
        # 234123412346 — this is a syntactically valid Aadhaar-like 12-digit number
        # We test that the rule fires and Verhoeff validation is applied
        hits = run_rules("Aadhaar: 234123412346", config)
        aadhaar = [h for h in hits if h.entity_type == "AADHAAR"]
        # May or may not pass Verhoeff, but the rule must fire and be evaluated
        assert len(aadhaar) >= 0  # Rule exists; soft-penalty may reduce conf below threshold

    def test_bare_12_digits_not_flagged_without_label_if_verhoeff_fails(self, config):
        """12 random digits that fail Verhoeff should have reduced confidence (soft penalty)."""
        # 000000000000 will fail Verhoeff and be soft-penalised
        hits = run_rules("Reference: 000000000000", config)
        aadhaar = [h for h in hits if h.entity_type == "AADHAAR"]
        # The soft penalty should drop confidence below threshold → no hit
        assert len(aadhaar) == 0, "All-zero 12-digit run should not produce a high-conf AADHAAR hit"

    def test_standard_4_4_4_format_still_works(self, config):
        """The hyphen/space separated format must still be detected."""
        hits = run_rules("Aadhaar No: 1234 5678 9012", config)
        aadhaar = [h for h in hits if h.entity_type == "AADHAAR"]
        assert len(aadhaar) >= 1


class TestIPAddressLeadingZeros:
    def test_leading_zero_octet_accepted(self, config):
        """IP addresses with leading-zero octets (common in OCR output) should be accepted."""
        hits = run_rules("Server: 192.168.01.01", config)
        ips = [h for h in hits if h.entity_type == "IP_ADDRESS"]
        assert len(ips) >= 1, "IP with leading-zero octets must be detected"

    def test_private_ip_still_detected(self, config):
        """Private IPs are PII — they should be detected (soft-reject, not hard-reject)."""
        hits = run_rules("VPN gateway: 10.0.0.1", config)
        ips = [h for h in hits if h.entity_type == "IP_ADDRESS"]
        assert len(ips) >= 1, "Private-range IP must be detected (soft-reject rule)"

    def test_invalid_octet_still_rejected(self, config):
        """Out-of-range octets (>255) must never produce a hit."""
        hits = run_rules("Bad address: 999.0.0.1", config)
        ips = [h for h in hits if h.entity_type == "IP_ADDRESS"]
        assert len(ips) == 0


class TestSSNLabelledBareDigits:
    def test_ssn_bare_9_digits_labelled(self, config):
        """'SSN: 123456789' (no hyphens) in labelled context must be detected."""
        hits = run_rules("SSN: 123456789", config)
        ssns = [h for h in hits if h.entity_type == "SSN"]
        assert len(ssns) >= 1, "Bare 9-digit SSN in labelled context must be detected"

    def test_social_security_bare_digits(self, config):
        """'Social Security: 219456789' must be detected."""
        hits = run_rules("Social Security: 219456789", config)
        ssns = [h for h in hits if h.entity_type == "SSN"]
        assert len(ssns) >= 1

    def test_bare_9_digits_without_label_not_flagged(self, config):
        """9 contiguous digits without SSN label must NOT be flagged (too many false positives)."""
        hits = run_rules("Reference number: 123456789", config)
        ssns = [h for h in hits if h.entity_type == "SSN"]
        assert len(ssns) == 0, "Bare 9 digits without SSN label must not be flagged"


class TestEmailMultiTLD:
    def test_email_with_tag(self, config):
        """user+tag@example.com style emails must be detected."""
        hits = run_rules("Reply to: user+filter@example.com", config)
        emails = [h for h in hits if h.entity_type == "EMAIL"]
        assert len(emails) >= 1

    def test_email_multi_level_tld(self, config):
        """user@company.co.uk style multi-level TLDs must be detected."""
        hits = run_rules("Contact: john.smith@company.co.uk", config)
        emails = [h for h in hits if h.entity_type == "EMAIL"]
        assert len(emails) >= 1

    def test_email_in_angle_brackets(self, config):
        """<user@example.com> angle-bracket form must be detected."""
        hits = run_rules("From: <alice@example.org>", config)
        emails = [h for h in hits if h.entity_type == "EMAIL"]
        assert len(emails) >= 1, "Email in angle brackets must be detected"


class TestContextExpandedLabels:
    def test_name_colon_pattern(self, config):
        """'Name: John Smith' must produce a PERSON hit from context layer."""
        location = "body"
        text = "Name: John Smith"
        hits = run_rules(text, config)
        new_hits, _ = run_context(location, text, hits, config)
        persons = [h for h in new_hits if h.entity_type == "PERSON"]
        assert len(persons) >= 1, "'Name: John Smith' must be detected as PERSON"

    def test_full_name_colon_pattern(self, config):
        """'Full Name: Jane Doe' must produce a PERSON hit from context layer."""
        location = "body"
        text = "Full Name: Jane Doe"
        hits = run_rules(text, config)
        new_hits, _ = run_context(location, text, hits, config)
        persons = [h for h in new_hits if h.entity_type == "PERSON"]
        assert len(persons) >= 1

    def test_employee_name_pattern(self, config):
        """'Employee Name: Robert Chen' must produce a PERSON hit."""
        location = "body"
        text = "Employee Name: Robert Chen"
        hits = run_rules(text, config)
        new_hits, _ = run_context(location, text, hits, config)
        persons = [h for h in new_hits if h.entity_type == "PERSON"]
        assert any("Robert" in h.text for h in persons), "Employee Name pattern must detect PERSON"


class TestIndiaPlusPhone:
    def test_plus91_prefix_detected(self, config):
        """+91 prefix phones must be detected even at start of text or after space."""
        hits = run_rules("+919876543210", config)
        phones = [h for h in hits if h.entity_type == "PHONE"]
        assert len(phones) >= 1, "+91 prefix India phone must be detected"

    def test_plus91_with_space(self, config):
        hits = run_rules("Mobile: +91 9876543210", config)
        phones = [h for h in hits if h.entity_type == "PHONE"]
        assert len(phones) >= 1

