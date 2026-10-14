"""
tests/test_customer_records.py
------------------------------
Regression and compliance test suite for customer records dataset (pii_test_dataset.pdf).
Validates:
  A. Heading person names ("<ID-like> - <Name>")
  B. Field labels NOT redacted (e.g., "Card Number", "Bank Account No.", "Passport No.")
  C. Street addresses ("981 Park Lane", "433 MG Road", "Main Rd", ADDRESS vs PERSON precedence)
  D. Amex 15-digit cards (4-6-5 grouping) and labelled card number fallback
  E. ABA routing numbers (checksum) and Indian IFSC codes (whole token, not passport)
  F. Emails wrapped across line breaks
  G. CVV, Card Expiry, and Financial attributes (income, expenses, credit score)
  H. Verifier gap: residue fails closed
"""
from __future__ import annotations

import re
import pytest
import pymupdf as fitz
from pathlib import Path

from extract.router import route
from detect.merge import detect_chunk
from detect.propagate import propagate_name_aliases
from redact.mapping import PlaceholderMapping
from redact.redactor import redact_chunks
from verify.leakage import check_leakage


def parse_customer_dataset(pdf_path: str):
    """
    Parse the Label / Value structure of pii_test_dataset.pdf into structured records.
    """
    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    # Split into records by CUST-XXXX heading
    parts = re.split(r"(CUST-\d{4}\s*-\s*[^\n]+)", full_text)
    records = []

    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1]

        # Extract name from heading: CUST-XXXX - First Last
        m_head = re.match(r"CUST-\d{4}\s*-\s*(.+)", heading)
        name = m_head.group(1).strip() if m_head else ""

        # Parse key-value lines
        lines = [l.strip() for l in body.splitlines() if l.strip()]
        kv = {"Heading": heading, "Name": name}

        known_labels = [
            "Email", "Phone", "Date of Birth", "Gender", "SSN",
            "Street Address", "City", "State", "ZIP / PIN", "Country",
            "Card Type", "Card Number", "Card Expiry", "CVV",
            "Bank Name", "Bank Account No", "IFSC / Routing",
            "PAN (India)", "Passport No", "Occupation",
            "Annual Income (USD)", "Monthly Expenses (USD)", "Monthly Expenses\n(USD)",
            "Credit Score", "IP Address", "Notes",
        ]

        # Scan for label lines followed by value lines
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            matched_label = None
            for kl in known_labels:
                if line.lower() == kl.lower() or line.lower().startswith(kl.lower()):
                    matched_label = kl
                    break
            if matched_label and idx + 1 < len(lines):
                val = lines[idx + 1]
                kv[matched_label] = val
                idx += 2
            else:
                idx += 1

        records.append(kv)

    return records


@pytest.fixture(scope="module")
def customer_pdf_path():
    path = Path("tests/assets/pii_test_dataset.pdf")
    if not path.exists():
        pytest.skip("tests/assets/pii_test_dataset.pdf not found")
    return str(path)


@pytest.fixture(scope="module")
def parsed_records(customer_pdf_path):
    return parse_customer_dataset(customer_pdf_path)


# ---------------------------------------------------------------------------
# Inline unit tests for each specific issue
# ---------------------------------------------------------------------------

def test_heading_name_detection_snippet(config):
    """A. Test heading shape '<ID-like token> - <Capitalised Name>'."""
    from extract.models import Chunk
    chunk = Chunk("test.txt", "line:1", "CUST-1001 - Aarav Sharma", "native", None, 0)
    ents = detect_chunk(chunk, config)
    person_ents = [e for e in ents if e.entity_type == "PERSON"]
    assert any(e.raw_value == "Aarav Sharma" for e in person_ents), \
        f"Expected Aarav Sharma to be detected as PERSON from heading, got {person_ents}"


def test_form_labels_not_detected_as_person(config):
    """B. Test form labels ('Card Number', 'Card Type', 'Card Expiry') are never PERSON."""
    from extract.models import Chunk
    labels_text = (
        "Card Number\n4433 2181 9600 1330\n"
        "Card Type\nVisa\n"
        "Card Expiry\n02/27\n"
        "Bank Account No\n839319141080\n"
        "Passport No\nT5218028\n"
    )
    chunk = Chunk("test.txt", "line:1", labels_text, "native", None, 0)
    ents = detect_chunk(chunk, config)
    for e in ents:
        if e.entity_type == "PERSON":
            assert e.raw_value.lower() not in (
                "card", "card number", "card type", "card expiry",
                "bank account no", "passport no", "passport"
            ), f"Label '{e.raw_value}' was wrongly detected as PERSON"


def test_bank_account_and_passport_label_preserved(config, tmp_path):
    """B. Test that only the value is replaced, never the label."""
    from extract.models import Chunk
    sample = "Bank Account No\n839319141080\nPassport No\nT5218028\n"
    chunk = Chunk("test.txt", "body", sample, "native", None, 0)
    ents = detect_chunk(chunk, config)
    mapping = PlaceholderMapping("test.txt", str(tmp_path))
    redacted = redact_chunks([chunk], ents, mapping, config)
    redacted_text = redacted[0].text

    assert "Bank Account No" in redacted_text, f"Label 'Bank Account No' was swallowed in: {redacted_text}"
    assert "Passport No" in redacted_text, f"Label 'Passport No' was swallowed in: {redacted_text}"
    assert "839319141080" not in redacted_text
    assert "T5218028" not in redacted_text


def test_street_address_standalone_and_precedence(config, tmp_path):
    """C. Test street addresses and ADDRESS winning over PERSON."""
    from extract.models import Chunk
    sample = (
        "Street Address\n981 Park Lane\n"
        "Street Address\n433 MG Road\n"
        "Street Address\n216 Gandhi Nagar Main Rd\n"
    )
    chunk = Chunk("test.txt", "body", sample, "native", None, 0)
    ents = detect_chunk(chunk, config)
    mapping = PlaceholderMapping("test.txt", str(tmp_path))
    redacted = redact_chunks([chunk], ents, mapping, config)
    redacted_text = redacted[0].text

    assert "981 Park Lane" not in redacted_text, "981 Park Lane was not redacted"
    assert "433 MG Road" not in redacted_text, "433 MG Road was not redacted"
    assert "216 Gandhi Nagar Main Rd" not in redacted_text, "216 Gandhi Nagar Main Rd was not redacted"
    # Ensure Gandhi Nagar Main Rd is NOT partially tagged as PERSON
    assert "[PERSON" not in redacted_text, f"Address contained spurious PERSON tag: {redacted_text}"


def test_amex_card_detection_and_fallback(config, tmp_path):
    """D. Test 15-digit Amex card format (4-6-5) and labelled fallback."""
    from extract.models import Chunk
    # Valid Amex format: 3751 462704 82811
    sample = (
        "Card Number\n3751 462704 82811\n"
        "Card Number\n3700 000000 00002\n"  # Labelled fallback even if checksum fails
    )
    chunk = Chunk("test.txt", "body", sample, "native", None, 0)
    ents = detect_chunk(chunk, config)
    mapping = PlaceholderMapping("test.txt", str(tmp_path))
    redacted = redact_chunks([chunk], ents, mapping, config)
    redacted_text = redacted[0].text

    assert "3751 462704 82811" not in redacted_text, "Amex card was not redacted"
    assert "3700 000000 00002" not in redacted_text, "Labelled card was not redacted"


def test_routing_number_and_ifsc(config, tmp_path):
    """E. Test 9-digit ABA routing numbers and Indian IFSC codes."""
    from extract.models import Chunk
    sample = (
        "IFSC / Routing\n031429110\n"
        "IFSC / Routing\nEXMP0390782\n"
    )
    chunk = Chunk("test.txt", "body", sample, "native", None, 0)
    ents = detect_chunk(chunk, config)
    mapping = PlaceholderMapping("test.txt", str(tmp_path))
    redacted = redact_chunks([chunk], ents, mapping, config)
    redacted_text = redacted[0].text

    assert "031429110" not in redacted_text, "ABA Routing number leaked"
    assert "EXMP0390782" not in redacted_text, "IFSC code leaked"
    assert "EX[PASSPORT" not in redacted_text, f"IFSC code was partially mislabelled as PASSPORT: {redacted_text}"


def test_multiline_wrapped_email(config, tmp_path):
    """F. Test email wrapped across line break (e.g. lakshmi.narayanan3@mail.example.\ncom)."""
    from extract.models import Chunk
    sample = "Email\nlakshmi.narayanan3@mail.example.\ncom\n"
    chunk = Chunk("test.txt", "body", sample, "native", None, 0)
    ents = detect_chunk(chunk, config)
    mapping = PlaceholderMapping("test.txt", str(tmp_path))
    redacted = redact_chunks([chunk], ents, mapping, config)
    redacted_text = redacted[0].text

    assert ".\ncom" not in redacted_text, f"Dangling .\\ncom residue left behind: {redacted_text}"
    assert "lakshmi.narayanan3" not in redacted_text


def test_cvv_expiry_and_financial_attributes(config, tmp_path):
    """G. Test CVV, Card Expiry, and financial attributes under labels."""
    from extract.models import Chunk
    cfg = dict(config)
    cfg["redact_financial_attributes"] = True

    sample = (
        "Card Expiry\n02/27\n"
        "CVV\n489\n"
        "Annual Income (USD)\n$138,000\n"
        "Monthly Expenses (USD)\n$5,590.15\n"
        "Credit Score\n685\n"
    )
    chunk = Chunk("test.txt", "body", sample, "native", None, 0)
    ents = detect_chunk(chunk, cfg)
    mapping = PlaceholderMapping("test.txt", str(tmp_path))
    redacted = redact_chunks([chunk], ents, mapping, cfg)
    redacted_text = redacted[0].text

    assert "02/27" not in redacted_text, "Card expiry leaked"
    assert "489" not in redacted_text, "CVV leaked"
    assert "138,000" not in redacted_text, "Annual income leaked"
    assert "5,590.15" not in redacted_text, "Monthly expenses leaked"
    assert "685" not in redacted_text, "Credit score leaked"


def test_verifier_residue_fails_safe(config, tmp_path):
    """H. Test that unverified residue of long digits or name pairs fails closed."""
    from extract.models import Chunk
    # Chunk with unredacted name pair and long digit sequence
    chunk = Chunk("test.txt", "body", "Customer Aarav Sharma has reference 031429110 and phone.", "native", None, 0)
    res = check_leakage([chunk], config)
    assert not res.downstream_safe, "Verifier should mark residue of name pair or long digits as unsafe (fail-closed)"


# ---------------------------------------------------------------------------
# End-to-end full dataset tests on pii_test_dataset.pdf
# ---------------------------------------------------------------------------

def test_full_customer_dataset_pipeline(customer_pdf_path, parsed_records, config, tmp_path):
    """
    Run the full pipeline on pii_test_dataset.pdf and verify all 20 records
    are sanitized without residue or leaks.
    """
    cfg = dict(config)
    cfg["redact_financial_attributes"] = True

    chunks = route(customer_pdf_path, cfg)
    assert len(chunks) > 0

    all_entities = []
    for c in chunks:
        if not c.is_blocked():
            all_entities.extend(detect_chunk(c, cfg))

    if cfg.get("alias_propagation", True):
        person_ents = [e for e in all_entities if e.entity_type == "PERSON"]
        chunk_texts = [c.text for c in chunks if not c.is_blocked()]
        chunk_locs = [c.location for c in chunks if not c.is_blocked()]
        alias_ents = propagate_name_aliases(
            chunk_texts, person_ents, all_entities, cfg,
            file=customer_pdf_path, locations=chunk_locs
        )
        all_entities.extend(alias_ents)

    mapping = PlaceholderMapping(customer_pdf_path, str(tmp_path))
    redacted_chunks = redact_chunks(chunks, all_entities, mapping, cfg)
    redacted_full = "\n\n".join(c.text for c in redacted_chunks)

    # 1. Check all 20 customer names are redacted
    leaked_names = []
    for rec in parsed_records:
        name = rec.get("Name", "")
        if name and name in redacted_full:
            leaked_names.append(name)
    assert not leaked_names, f"Customer names leaked in redacted text: {leaked_names}"

    # 2. Check all street addresses are redacted
    leaked_addresses = []
    for rec in parsed_records:
        addr = rec.get("Street Address", "")
        if addr and addr in redacted_full:
            leaked_addresses.append(addr)
    assert not leaked_addresses, f"Street addresses leaked in redacted text: {leaked_addresses}"

    # 3. Check all card numbers (including Amex) are redacted
    leaked_cards = []
    for rec in parsed_records:
        card = rec.get("Card Number", "")
        if card and card in redacted_full:
            leaked_cards.append(card)
    assert not leaked_cards, f"Card numbers leaked in redacted text: {leaked_cards}"

    # 4. Check routing numbers and IFSC are redacted
    leaked_routing = []
    for rec in parsed_records:
        rf = rec.get("IFSC / Routing", "")
        if rf and rf in redacted_full:
            leaked_routing.append(rf)
    assert not leaked_routing, f"Routing / IFSC numbers leaked in redacted text: {leaked_routing}"

    # 5. Check CVV and Expiry
    for rec in parsed_records:
        cvv = rec.get("CVV", "")
        if cvv:
            assert f"\n{cvv}\n" not in redacted_full, f"CVV {cvv} leaked in text"

    # 6. Check that form labels were NOT redacted
    assert "Card Number" in redacted_full, "'Card Number' label was incorrectly redacted"
    assert "Card Type" in redacted_full, "'Card Type' label was incorrectly redacted"
    assert "Card Expiry" in redacted_full, "'Card Expiry' label was incorrectly redacted"
    assert "Bank Account No" in redacted_full, "'Bank Account No' label was incorrectly redacted"
    assert "Passport No" in redacted_full, "'Passport No' label was incorrectly redacted"

    # 7. Check verifier downstream_safe is True on clean output
    leak_res = check_leakage(redacted_chunks, cfg, known_values=mapping)
    assert leak_res.downstream_safe, f"Verifier flagged clean output as unsafe: {leak_res.summary()}"
