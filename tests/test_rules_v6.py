"""
tests/test_rules_v6.py
----------------------
Unit test suite verifying Phase 6 fixes for:
- Issue 1: International phone numbers
- Issue 2: OCR-tolerant email precision (no false positives on ordinary words)
- Issue 3: SSN 9xx in narrative context and last-four redaction
- Issue 4: DATE_OF_BIRTH detection (and rejection of ordinary document dates)
- Issue 5: ADDRESS detection (and corporate address exclusion)
- Issue 7: Extra identifiers (DIRECTOR_ID, VENDOR_ID, PESEL, account ending)
- Issue 8: Label word boundaries and look-alike guards (no false positives on distinguishing, contingency, etc.)
"""
import pytest
import yaml
from pathlib import Path

from detect.rules import run_rules


@pytest.fixture
def config():
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestIssue1InternationalPhones:
    @pytest.mark.parametrize("phone,expected_clean", [
        ("+44 20 7555 0171", "+44 20 7555 0171"),
        ("+44 20 75550171", "+44 20 75550171"),
        ("+44 7700 555 0163 (mobile)", "+44 7700 555 0163"),
        ("+91 80 4555 0182", "+91 80 4555 0182"),
        ("+91 22 4555 0117", "+91 22 4555 0117"),
        ("+91 98860 55107", "+91 98860 55107"),
        ("+33 1 5555 0144", "+33 1 5555 0144"),
        ("+33 15555 0144", "+33 15555 0144"),
        ("+65 6555 0172", "+65 6555 0172"),
        ("+49 69 5555 0177", "+49 69 5555 0177"),
        ("+46 8 5555 0121", "+46 8 5555 0121"),
        ("+46 70 555 0188", "+46 70 555 0188"),
        ("+48 22 555 0164", "+48 22 555 0164"),
        ("+48 512 555 0180", "+48 512 555 0180"),
        ("+63 917 555 0119", "+63 917 555 0119"),
        ("+1 (646) 555-0203", "+1 (646) 555-0203"),
        ("+1 212 555-0147", "+1 212 555-0147"),
    ])
    def test_international_phones_matched(self, config, phone, expected_clean):
        hits = run_rules(f"Contact us at {phone} for immediate assistance.", config)
        phone_hits = [h for h in hits if h.entity_type == "PHONE"]
        assert len(phone_hits) == 1, f"Failed for {phone}: got {phone_hits}"
        assert phone_hits[0].text == expected_clean
        assert "(mobile)" not in phone_hits[0].text

    def test_phone_exclusions_not_matched(self, config):
        non_phones = [
            "Effective date: 2026-04-12",
            "Software version: v5.4.1",
            "Control specification: QRC_Sec.9881",
            "Completed in 35 days",
        ]
        for text in non_phones:
            hits = run_rules(text, config)
            phone_hits = [h for h in hits if h.entity_type == "PHONE"]
            assert len(phone_hits) == 0, f"False positive on {text}: {phone_hits}"


class TestIssue2EmailPrecision:
    def test_ordinary_words_not_flagged_as_email(self, config):
        sentences = [
            "Consider what could go wrong before deploying.",
            "Structuring safeguards so that risk-taking is deliberate.",
            "The security audit was a great success.",
            "Maintain at least three redundant clusters.",
            "Controls must be set at an appropriate level.",
        ]
        for s in sentences:
            # Test with is_ocr=True as well
            hits = run_rules(s, config, is_ocr=True)
            email_hits = [h for h in hits if h.entity_type == "EMAIL"]
            assert len(email_hits) == 0, f"False positive EMAIL in: {s} -> {email_hits}"

    def test_real_and_tolerant_emails_detected(self, config):
        real_emails = [
            "Contact user@example.com for help.",
            "Write to admin (at) cadencecorp.net for questions.",
            "Escalate to lead [at] cadencecorp.net immediately.",
            "Reach security at cadencecorp.net for incidents.",
        ]
        for e in real_emails:
            hits = run_rules(e, config, is_ocr=True)
            email_hits = [h for h in hits if h.entity_type == "EMAIL"]
            assert len(email_hits) == 1, f"Failed to detect email in: {e}"


class TestIssue3SSNAndLastFour:
    def test_narrative_9xx_ssn_detected(self, config):
        texts = [
            ("Investigation file records SSN 900-55-2043 during intake.", "900-55-2043"),
            ("A cross-reference audit was initiated showing SSN 900-42-7781 in the register.", "900-42-7781"),
        ]
        for text, val in texts:
            hits = run_rules(text, config)
            ssn_hits = [h for h in hits if h.entity_type == "SSN"]
            assert any(val in h.text for h in ssn_hits), f"Failed for {text}"

    def test_last_four_ssn_detected(self, config):
        text = "Please confirm the last four digits of your social security number (8827) before access."
        hits = run_rules(text, config)
        ssn_hits = [h for h in hits if h.entity_type == "SSN"]
        assert len(ssn_hits) >= 1
        assert any("8827" in h.text for h in ssn_hits)

    def test_card_account_ending_detected(self, config):
        texts = [
            ("Client reported compromised card ending 2218 on Monday.", "2218"),
            ("Wire confirmation for savings account ending 7741 yesterday.", "7741"),
        ]
        for text, val in texts:
            hits = run_rules(text, config)
            acct_hits = [h for h in hits if h.entity_type == "ACCOUNT_NUMBER"]
            assert len(acct_hits) >= 1
            assert any(val in h.text for h in acct_hits)


class TestIssue4DateOfBirth:
    def test_labelled_dob_detected(self, config):
        texts = [
            ("Candidate form lists DOB 14 July 1991 in the field.", "14 July 1991"),
            ("Participant verified with date of birth, 3 August 1996 on passport.", "3 August 1996"),
            ("Badge details:\nDATE OF BIRTH\n28 February 1979\nValid through 2030", "28 February 1979"),
        ]
        for text, val in texts:
            hits = run_rules(text, config)
            dob_hits = [h for h in hits if h.entity_type == "DATE_OF_BIRTH"]
            assert len(dob_hits) >= 1, f"Failed for {text}"
            assert any(val in h.text for h in dob_hits)

    def test_ordinary_document_dates_not_dob(self, config):
        ordinary = [
            "Effective Date: 2024-01-15",
            "Next policy review scheduled for 15 September 2026",
            "Version 3.2 release date: 2025-11-20",
        ]
        for text in ordinary:
            hits = run_rules(text, config)
            dob_hits = [h for h in hits if h.entity_type == "DATE_OF_BIRTH"]
            assert len(dob_hits) == 0, f"False positive DATE_OF_BIRTH on {text}: {dob_hits}"


class TestIssue5AddressDetector:
    def test_home_addresses_detected(self, config):
        addresses = [
            ("Official filing: home address 88 Willowbank Court, Hoboken, NJ 07030 on record.", "88 Willowbank Court, Hoboken, NJ 07030"),
            ("Employee updated his home address, 1180 Ashwood Lane, Apartment 4C, Jersey City, NJ 07302 last month.", "1180 Ashwood Lane, Apartment 4C, Jersey City, NJ 07302"),
            ("Notice dispatched to your home at 2214 Palmetto Crescent, Coral Gables, FL 33134 via courier.", "2214 Palmetto Crescent, Coral Gables, FL 33134"),
        ]
        for text, val in addresses:
            hits = run_rules(text, config)
            addr_hits = [h for h in hits if h.entity_type == "ADDRESS"]
            assert len(addr_hits) >= 1, f"Failed for {text}"
            assert any(val in h.text or h.text in val for h in addr_hits)

    def test_corporate_address_excluded(self, config):
        text = "Corporate HQ: Cadence Financial Group, 100 Wall Street, New York, NY 10005"
        hits = run_rules(text, config)
        addr_hits = [h for h in hits if h.entity_type == "ADDRESS"]
        assert len(addr_hits) == 0, f"Corporate address was flagged: {addr_hits}"


class TestIssue7ExtraIdentifiers:
    def test_extra_identifiers_detected(self, config):
        cases = [
            ("Director sign-off by DIR-0042 required.", "DIRECTOR_ID", "DIR-0042"),
            ("Vendor contracting under MER-IN-4471 active.", "VENDOR_ID", "MER-IN-4471"),
            ("Vendor regional code MER-PL-1187 verified.", "VENDOR_ID", "MER-PL-1187"),
            ("Vendor billing under MER-PH-2214 processed.", "VENDOR_ID", "MER-PH-2214"),
            ("Polish tax file PESEL 890412xxxxx in registry.", "PESEL", "890412xxxxx"),
        ]
        for text, etype, val in cases:
            hits = run_rules(text, config)
            matches = [h for h in hits if h.entity_type == etype]
            assert len(matches) >= 1, f"Failed {etype} in {text}"
            assert any(val in h.text for h in matches)


class TestIssue8LabelWordsPrecision:
    def test_label_words_not_flagged(self, config):
        texts = [
            "All employee identifiers must be logged.",
            "All national identifiers must be protected.",
            "Reviewing distinguishing features of the system.",
            "Execute the contingency protocol.",
            "Reviewed by Board Risk Committee.",
            "Dispatched by Data Privacy Office.",
            "Table column header: NATIONAL ID CONTACT",
        ]
        for text in texts:
            hits = run_rules(text, config)
            # Must NOT produce GOVERNMENT_ID or EMPLOYEE_ID
            bad_hits = [h for h in hits if h.entity_type in ("GOVERNMENT_ID", "EMPLOYEE_ID")]
            assert len(bad_hits) == 0, f"False positive ID in {text}: {bad_hits}"

    def test_aadhaar_date_phone_lookalike_not_flagged(self, config):
        text = "Drafted 21 Apr 2026 +44 20 7555 0171"
        hits = run_rules(text, config)
        aadhaar_hits = [h for h in hits if h.entity_type == "AADHAAR"]
        assert len(aadhaar_hits) == 0, f"False positive AADHAAR on {text}: {aadhaar_hits}"
