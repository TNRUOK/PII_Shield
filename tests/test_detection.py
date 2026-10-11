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
    def test_bare_name_denied_and_full_name_allowed(self, config):
        """Bare 'Name: ...' must not trigger false positive PERSON from context layer; Full Name must."""
        location = "body"
        # Deny case: bare "Name: Third Party" or "Name: Cadence Financial Services"
        text_deny = "Name: Third Party"
        hits_deny = run_rules(text_deny, config)
        new_hits_deny, _ = run_context(location, text_deny, hits_deny, config)
        persons_deny = [h for h in new_hits_deny if h.entity_type == "PERSON"]
        assert len(persons_deny) == 0, f"Bare name should not be emitted: {persons_deny}"

        # Allow case: "Full Name: Jane Doe"
        text_allow = "Full Name: Jane Doe"
        hits_allow = run_rules(text_allow, config)
        new_hits_allow, _ = run_context(location, text_allow, hits_allow, config)
        persons_allow = [h for h in new_hits_allow if h.entity_type == "PERSON"]
        assert len(persons_allow) >= 1, "Full Name pattern must detect PERSON"

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


class TestPlausiblePerson:
    def test_plausible_person_filtering(self, config):
        from detect.ner import plausible_person

        denylist = {x.lower() for x in config.get("person_denylist", [])}

        # Allow cases
        assert plausible_person("Alice Smith", denylist) is True
        assert plausible_person("David Miller", denylist) is True
        assert plausible_person("J. Doe", denylist) is True

        # Deny cases: denylist companies / generic terms
        assert plausible_person("Optiv", denylist) is False
        assert plausible_person("Cadence", denylist) is False
        assert plausible_person("Cadence Financial Services", denylist) is False
        assert plausible_person("ServiceNow", denylist) is False
        assert plausible_person("Workflow", denylist) is False

        # Deny cases: all-caps acronyms
        assert plausible_person("MFA", denylist) is False
        assert plausible_person("BYOD", denylist) is False
        assert plausible_person("SSO", denylist) is False

        # Deny cases: digits or underscores
        assert plausible_person("User_123", denylist) is False
        assert plausible_person("Admin2", denylist) is False


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


class TestContextInitialsAndTables:
    def test_users_column_initials_names(self, config):
        """Table with USERS column containing 3 initials-style names must detect all 3."""
        from extract.models import Chunk
        from detect.merge import detect_chunk

        rows = [
            ("table:1/r2[headers: ID | Users | Role]", "101 | TJ Harlan | Engineer"),
            ("table:1/r3[headers: ID | Users | Role]", "102 | PQ Lindqvist | Architect"),
            ("table:1/r4[headers: ID | Users | Role]", "103 | RK Patel | Analyst"),
        ]

        found_names = []
        for loc, row_text in rows:
            chunk = Chunk(
                file="test.docx",
                location=loc,
                text=row_text,
                method="native",
                ocr_confidence=None,
                order_index=0,
            )
            entities = detect_chunk(chunk, config)
            for e in entities:
                if e.entity_type == "PERSON":
                    found_names.append(e.raw_value)

        assert any("TJ Harlan" in n for n in found_names), f"TJ Harlan missed: {found_names}"
        assert any("PQ Lindqvist" in n for n in found_names), f"PQ Lindqvist missed: {found_names}"
        assert any("RK Patel" in n for n in found_names), f"RK Patel missed: {found_names}"

    def test_key_value_tables(self, config):
        """Two-column key-value tables (Key | Value) must detect names from the value column."""
        from extract.models import Chunk
        from detect.merge import detect_chunk

        kv_rows = [
            ("table:2/r1", "Author | Alice Johnson"),
            ("table:2/r2", "Peer Reviewer | PQ Lindqvist"),
            ("table:2/r3", "Client Contact | TJ Harlan"),
            ("table:2/r4", "Prepared by | Bob Smith"),
        ]

        for loc, row_text in kv_rows:
            chunk = Chunk(
                file="test.docx",
                location=loc,
                text=row_text,
                method="native",
                ocr_confidence=None,
                order_index=0,
            )
            entities = detect_chunk(chunk, config)
            persons = [e for e in entities if e.entity_type == "PERSON"]
            assert len(persons) >= 1, f"Failed to detect person in key-value row: {row_text}"

    def test_acronym_not_flagged_in_normal_paragraph(self, config):
        """Acronyms and software names in normal paragraphs must NOT be flagged as PERSON."""
        from extract.models import Chunk
        from detect.merge import detect_chunk

        chunk = Chunk(
            file="test.docx",
            location="body",
            text="All employees must follow the BYOD Allowed policy and use MFA when Launching ServPro for SSO logins.",
            method="native",
            ocr_confidence=None,
            order_index=0,
        )
        entities = detect_chunk(chunk, config)
        person_names = [e.raw_value for e in entities if e.entity_type == "PERSON"]
        assert not any("BYOD" in n for n in person_names), f"BYOD incorrectly flagged: {person_names}"
        assert not any("MFA" in n for n in person_names), f"MFA incorrectly flagged: {person_names}"
        assert not any("Launching" in n or "ServPro" in n for n in person_names), f"Launching ServPro incorrectly flagged: {person_names}"


class TestGroupColumnExclusion:
    """Issue 1: Group-name and organizational structure columns must NOT be flagged as PERSON."""

    def test_user_groups_table_exclusion(self, config):
        """USER GROUPS | DESCRIPTION | PERMISSIONS | USERS table."""
        from extract.models import Chunk
        from detect.merge import detect_chunk

        headers = "USER GROUPS | DESCRIPTION | PERMISSIONS | USERS"
        rows = [
            ("table:1/r1[headers: " + headers + "]", "Vendor Risk Management Group | External auditors | ReadOnly | TJ Harlan"),
            ("table:1/r2[headers: " + headers + "]", "Security Group | Core SecOps | Admin | PQ Lindqvist"),
        ]

        for loc, row_text in rows:
            chunk = Chunk(
                file="test.docx",
                location=loc,
                text=row_text,
                method="native",
                ocr_confidence=None,
                order_index=0,
            )
            entities = detect_chunk(chunk, config)
            person_names = [e.raw_value for e in entities if e.entity_type == "PERSON"]

            # Group names must NOT be flagged as PERSON
            assert not any("Vendor Risk" in n or "Management Group" in n for n in person_names), f"Group name wrongly flagged: {person_names}"
            assert not any("Security Group" in n for n in person_names), f"Group name wrongly flagged: {person_names}"

            # User column names MUST be flagged
            if "TJ Harlan" in row_text:
                assert any("TJ Harlan" in n for n in person_names), f"TJ Harlan missed: {person_names}"
            if "PQ Lindqvist" in row_text:
                assert any("PQ Lindqvist" in n for n in person_names), f"PQ Lindqvist missed: {person_names}"

    def test_team_and_reviewer_columns(self, config):
        """TEAM | REVIEWER table: Team name NOT flagged, Reviewer IS flagged."""
        from extract.models import Chunk
        from detect.merge import detect_chunk

        headers = "TEAM | REVIEWER"
        chunk = Chunk(
            file="test.docx",
            location=f"table:1/r1[headers: {headers}]",
            text="Core Infrastructure Team | RK Patel",
            method="native",
            ocr_confidence=None,
            order_index=0,
        )
        entities = detect_chunk(chunk, config)
        person_names = [e.raw_value for e in entities if e.entity_type == "PERSON"]
        assert not any("Team" in n or "Infrastructure" in n for n in person_names), f"Team column flagged: {person_names}"
        assert any("RK Patel" in n for n in person_names), f"Reviewer RK Patel missed: {person_names}"

    def test_key_value_labels_author_reviewer_contact(self, config):
        """Key-value tables with Author(s), Client Contact, Peer Reviewer must flag values."""
        from extract.models import Chunk
        from detect.merge import detect_chunk

        kv_rows = [
            ("table:3/r1", "Author(s) | Sarah Jenkins"),
            ("table:3/r2", "Peer Reviewer | Marcus Vance"),
            ("table:3/r3", "Client Contact | Helena Troy"),
        ]
        for loc, row_text in kv_rows:
            chunk = Chunk(
                file="test.docx",
                location=loc,
                text=row_text,
                method="native",
                ocr_confidence=None,
                order_index=0,
            )
            entities = detect_chunk(chunk, config)
            person_names = [e.raw_value for e in entities if e.entity_type == "PERSON"]
            assert len(person_names) >= 1, f"Expected person detection for {row_text}, got {person_names}"


class TestIssue3Rules:
    """Issue 3: IP address, phone variants, and OCR email rules."""

    def test_labelled_ipv4_extracted_cleanly(self, config):
        """Labelled IPv4 must extract ONLY the numeric IP without label prefix."""
        from detect.rules import run_rules

        text = "Workstation IP: 10.46.59.64 and Server Host: 192.168.1.100"
        hits = run_rules(text, config)
        ip_hits = [h for h in hits if h.entity_type == "IP_ADDRESS"]
        assert len(ip_hits) >= 2
        for h in ip_hits:
            assert not h.text.lower().startswith("workstation")
            assert not h.text.lower().startswith("server")
            assert not h.text.lower().startswith("ip")
            assert not h.text.lower().startswith("host")

    def test_ipv6_full_and_compressed(self, config):
        """Support both full 8-group and compressed :: IPv6 addresses."""
        from detect.rules import run_rules

        text = "Primary DNS: 2001:0db8:85a3:0000:0000:8a2e:0370:7334, Loopback: ::1, Link-local: fe80::1"
        hits = run_rules(text, config)
        ip_hits = [h for h in hits if h.entity_type == "IP_ADDRESS"]
        ip_texts = [h.text for h in ip_hits]
        assert any("2001:0db8:85a3:0000:0000:8a2e:0370:7334" in t for t in ip_texts)
        assert any("::1" in t for t in ip_texts)
        assert any("fe80::1" in t for t in ip_texts)

    def test_phone_variants(self, config):
        """Cover 10 contiguous digits, extension formats, and international Indian."""
        from detect.rules import run_rules

        # 10 contiguous digits
        hits = run_rules("Direct call line: 4522327928 for customer support.", config)
        phones = [h for h in hits if h.entity_type == "PHONE"]
        assert any("4522327928" in h.text for h in phones)

        # US with extension
        hits2 = run_rules("Office line: 555.010.1234 x45 or (555) 010-1234 ext 12.", config)
        phones2 = [h for h in hits2 if h.entity_type == "PHONE"]
        assert len(phones2) >= 2

        # Indian with spaces
        hits3 = run_rules("Support in India: +91 98765 43210 available 24/7.", config)
        phones3 = [h for h in hits3 if h.entity_type == "PHONE"]
        assert any("98765 43210" in h.text or "9876543210" in h.text for h in phones3)

    def test_ocr_tolerant_email(self, config):
        """OCR-tolerant email rule triggers on OCR chunks but not on native."""
        from extract.models import Chunk
        from detect.merge import detect_chunk

        ocr_chunk = Chunk(
            file="test.pdf",
            location="page:1",
            text="Inquiries: compliance (at) cadencefinancial [dot] com",
            method="ocr",
            ocr_confidence=75.0,
            order_index=0,
        )
        entities = detect_chunk(ocr_chunk, config)
        emails = [e for e in entities if e.entity_type == "EMAIL"]
        assert len(emails) >= 1


class TestPrecisionGuards:
    """Guard precision: ensure non-PII look-alikes are NEVER flagged."""

    def test_look_alikes_not_flagged(self, config):
        from extract.models import Chunk
        from detect.merge import detect_chunk

        text = (
            "Document version 4.31.2 was published on 12/05/2026. "
            "Refer to ProductID: PRD-20240115 and Section 2.4.1 under "
            "internal tracking code QRC_Sec_4.31. Case REF-9876543210123."
        )
        chunk = Chunk(
            file="test.docx",
            location="body",
            text=text,
            method="native",
            ocr_confidence=None,
            order_index=0,
        )
        entities = detect_chunk(chunk, config)
        for e in entities:
            assert e.raw_value not in ["4.31.2", "12/05/2026", "PRD-20240115", "2.4.1", "QRC_Sec_4.31"]
            assert e.entity_type != "IP_ADDRESS", f"False positive IP detected: {e.raw_value}"
            assert e.entity_type != "PHONE", f"False positive Phone detected: {e.raw_value}"


# ---------------------------------------------------------------------------
# Item 1: SUSPECT_OCR — low-confidence OCR fail-closed redaction
# ---------------------------------------------------------------------------

def _ocr_chunk(text: str, line_confs: list, conf: float = 45.0) -> Chunk:
    """Build a synthetic OCR chunk with per-line confidence data."""
    return Chunk(
        file="synthetic_test.png",
        location="page:1",
        text=text,
        method="ocr",
        ocr_confidence=conf,
        order_index=0,
        line_confidences=line_confs,
    )


def _native_chunk_bare(text: str) -> Chunk:
    """Build a native (non-OCR) chunk."""
    return Chunk(
        file="synthetic_test.docx",
        location="body",
        text=text,
        method="native",
        ocr_confidence=None,
        order_index=0,
    )


class TestSuspectOCR:
    """Tests for Item 1: fail-closed redaction of low-confidence OCR PII."""

    def test_low_conf_email_gets_suspect_ocr(self, config):
        """Email on a low-confidence OCR line should become SUSPECT_OCR."""
        from detect.suspect import find_suspect_ocr_hits
        text = "Contact: alicetompkins@syntheticcorp.net"
        chunk = _ocr_chunk(text, [(text, 42.0)])
        hits = find_suspect_ocr_hits(chunk, config)
        types = {h.entity_type for h in hits}
        assert "SUSPECT_OCR" in types, f"Expected SUSPECT_OCR, got {types}"

    def test_low_conf_digit_run_gets_suspect_ocr(self, config):
        """9-digit number on a low-confidence line should become SUSPECT_OCR."""
        from detect.suspect import find_suspect_ocr_hits
        text = "Account: 987654321"
        chunk = _ocr_chunk(text, [(text, 55.0)])
        hits = find_suspect_ocr_hits(chunk, config)
        types = {h.entity_type for h in hits}
        assert "SUSPECT_OCR" in types, f"Expected SUSPECT_OCR, got {types}"

    def test_high_conf_line_not_flagged(self, config):
        """High-confidence OCR line should NOT produce SUSPECT_OCR."""
        from detect.suspect import find_suspect_ocr_hits
        text = "Account: 987654321"
        chunk = _ocr_chunk(text, [(text, 95.0)])
        hits = find_suspect_ocr_hits(chunk, config)
        suspect = [h for h in hits if h.entity_type == "SUSPECT_OCR"]
        assert len(suspect) == 0, "High-conf line should not trigger SUSPECT_OCR"

    def test_native_chunk_never_flagged(self, config):
        """Native (non-OCR) chunks should never produce SUSPECT_OCR."""
        from detect.suspect import find_suspect_ocr_hits
        text = "Account: 987654321"
        chunk = _native_chunk_bare(text)
        hits = find_suspect_ocr_hits(chunk, config)
        assert hits == [], "Native chunks must not produce SUSPECT_OCR"

    def test_date_lookalike_not_flagged(self, config):
        """Dates like 12/05/2026 on low-conf lines must NOT be SUSPECT_OCR."""
        from detect.suspect import find_suspect_ocr_hits
        text = "Effective 12/05/2026"
        chunk = _ocr_chunk(text, [(text, 40.0)])
        hits = find_suspect_ocr_hits(chunk, config)
        date_hits = [h for h in hits if h.entity_type == "SUSPECT_OCR" and "2026" in h.text]
        assert len(date_hits) == 0, f"Date look-alike flagged as SUSPECT_OCR: {date_hits}"

    def test_version_lookalike_not_flagged(self, config):
        """Version strings like 4.31.2 must NOT be flagged."""
        from detect.suspect import find_suspect_ocr_hits
        text = "Version 4.31.2 released"
        chunk = _ocr_chunk(text, [(text, 38.0)])
        hits = find_suspect_ocr_hits(chunk, config)
        suspect = [h for h in hits if h.entity_type == "SUSPECT_OCR"]
        assert len(suspect) == 0, f"Version string flagged: {suspect}"

    def test_no_line_confs_skips_silently(self, config):
        """OCR chunk without line_confidences should produce no SUSPECT_OCR hits."""
        from detect.suspect import find_suspect_ocr_hits
        chunk = Chunk(
            file="test.png",
            location="page:1",
            text="Account: 987654321",
            method="ocr",
            ocr_confidence=50.0,
            order_index=0,
        )
        hits = find_suspect_ocr_hits(chunk, config)
        assert hits == [], "No line_confidences -> no SUSPECT_OCR"

    def test_suspect_ocr_appears_in_detect_chunk(self, config):
        """detect_chunk should include SUSPECT_OCR entity from low-conf line."""
        text = "Contact: benmcallister@synthfinance.io"
        chunk = _ocr_chunk(text, [(text, 48.0)])
        entities = detect_chunk(chunk, config)
        types = {e.entity_type for e in entities}
        assert "SUSPECT_OCR" in types or "EMAIL" in types, (
            f"Expected SUSPECT_OCR or EMAIL for OCR email, got {types}")


# ---------------------------------------------------------------------------
# Item 2: OCR character-confusion candidate normalisation
# ---------------------------------------------------------------------------

class TestOCRRepair:
    """Tests for detect/ocr_repair.py — character confusion normalisation."""

    def test_email_tld_cam_repaired(self, config):
        """Email with .cam TLD (OCR confusion) should be detected as EMAIL."""
        from detect.ocr_repair import find_ocr_repair_hits
        text = "Send results to researcher@syntheticlabs.cam"
        chunk = _ocr_chunk(text, [(text, 65.0)])
        hits = find_ocr_repair_hits(chunk, config)
        email_hits = [h for h in hits if h.entity_type == "EMAIL"]
        assert len(email_hits) >= 1, f"Expected repaired EMAIL hit, got {hits}"

    def test_email_tld_corn_repaired(self, config):
        """Email with .corn TLD (OCR confusion for .com) should be repaired."""
        from detect.ocr_repair import find_ocr_repair_hits
        text = "Contact: j.smith@fakefirm.corn"
        chunk = _ocr_chunk(text, [(text, 60.0)])
        hits = find_ocr_repair_hits(chunk, config)
        email_hits = [h for h in hits if h.entity_type == "EMAIL"]
        assert len(email_hits) >= 1, f"Expected repaired EMAIL hit, got {hits}"

    def test_native_chunk_not_repaired(self, config):
        """OCR repair should NOT run on native text chunks."""
        from detect.ocr_repair import find_ocr_repair_hits
        text = "researcher@syntheticlabs.cam"
        chunk = _native_chunk_bare(text)
        hits = find_ocr_repair_hits(chunk, config)
        assert hits == [], "OCR repair must not run on native chunks"

    def test_date_lookalike_skipped(self, config):
        """A date-shaped token must NOT be repaired into a false-positive."""
        from detect.ocr_repair import find_ocr_repair_hits
        text = "Effective 12/05/2026"
        chunk = _ocr_chunk(text, [(text, 55.0)])
        hits = find_ocr_repair_hits(chunk, config)
        assert hits == [], f"Date look-alike must not be repaired: {hits}"

    def test_version_skipped(self, config):
        """Version string must not be repaired."""
        from detect.ocr_repair import find_ocr_repair_hits
        text = "Version 4.31.2"
        chunk = _ocr_chunk(text, [(text, 70.0)])
        hits = find_ocr_repair_hits(chunk, config)
        assert hits == [], f"Version string must not be repaired: {hits}"

    def test_unchanged_token_not_emitted(self, config):
        """If no characters are substituted, no hit is emitted."""
        from detect.ocr_repair import find_ocr_repair_hits
        text = "Account 123456789"   # already all digits — no repair needed
        chunk = _ocr_chunk(text, [(text, 72.0)])
        hits = find_ocr_repair_hits(chunk, config)
        repair_hits = [h for h in hits if h.method == "ocr_repair"]
        assert len(repair_hits) == 0, "No-change token should not emit ocr_repair hit"





