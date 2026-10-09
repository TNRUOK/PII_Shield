"""
synthetic/generate.py
---------------------
Generate three synthetic documents with injected fake PII for testing.
Uses Faker for realistic-looking but entirely fake data.

Documents:
  1. scanned_policy.pdf  — image-only PDF (no text layer), ~6 pages
  2. training_guide.docx — rich DOCX with tables, header/footer, footnote, embedded PNG
  3. org_pack.pptx       — PPTX with directory table, RACI table, grouped shapes, notes

For each document:
  - Writes the document file to synthetic/
  - Writes ground_truth/<name>.json  listing every injected PII entity
  - Writes ground_truth/<stem>_text.txt with expected plain text (structure reference)

Tricky cases injected deliberately:
  - A name split across a line break (DOCX)
  - An UPPERCASE name
  - An ID that looks like a phone number (labelled so must still be detected as employee ID)
  - A number that FAILS Luhn (must NOT be flagged as credit card)
  - Company names and product names (false-positive traps)
  - A number formatted as 4-4-4 that fails Verhoeff (flagged at lower confidence)
"""
from __future__ import annotations

import io
import json
import math
import os
import random
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import fitz  # PyMuPDF
from faker import Faker
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pptx import Presentation
from pptx.util import Inches as PInches, Pt as PPt, Emu
from pptx.dml.color import RGBColor as PRGBColor
from pptx.enum.text import PP_ALIGN
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas as rl_canvas

fake = Faker("en_US")
Faker.seed(42)
random.seed(42)

OUT_DIR = Path(__file__).parent
GT_DIR = OUT_DIR / "ground_truth"
GT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_ssn() -> str:
    """Generate a fake but SSN-sanity-valid number."""
    while True:
        area = random.randint(1, 899)
        if area == 666:
            continue
        group = random.randint(1, 99)
        serial = random.randint(1, 9999)
        if area != 0 and group != 0 and serial != 0:
            return f"{area:03d}-{group:02d}-{serial:04d}"


def _fake_employee_id() -> str:
    return f"EMP-{random.randint(10000, 99999)}"


def _fake_aadhaar_invalid() -> str:
    """Return a 4-4-4 number that fails Verhoeff (for tricky-case injection)."""
    # Generate random and guarantee it fails by flipping the last digit
    digits = [str(random.randint(0, 9)) for _ in range(12)]
    digits[-1] = str((int(digits[-1]) + 1) % 10)
    d = "".join(digits)
    return f"{d[:4]} {d[4:8]} {d[8:]}"


def _fake_non_luhn_card() -> str:
    """A plausible-looking card number that fails Luhn — should NOT be flagged."""
    return "4111 1111 1111 1112"  # last digit wrong → fails Luhn


def _fake_ip() -> str:
    return f"10.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"


def _save_gt(filename: str, entities: List[Dict], doc_name: str) -> None:
    path = GT_DIR / filename
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"document": doc_name, "entities": entities}, fh, indent=2)
    print(f"  Ground truth → {path}")


def _save_gt_text(filename: str, text: str) -> None:
    path = GT_DIR / filename
    path.write_text(text, encoding="utf-8")
    print(f"  GT text      → {path}")


def _gt_entity(
    etype: str, text: str, location: str, tricky: bool = False
) -> Dict:
    return {"type": etype, "text": text, "location": location, "tricky": tricky}


# ---------------------------------------------------------------------------
# 1. scanned_policy.pdf (image-only PDF)
# ---------------------------------------------------------------------------

def _render_page_image(text_lines: List[str], rotate_deg: float = 0.0,
                        noise_level: int = 5) -> Image.Image:
    """
    Render text lines onto a white page as a PIL image.
    Optionally apply small rotation and Gaussian-ish noise.
    """
    W, H = 794, 1123  # A4 at 96 DPI approx
    img = Image.new("RGB", (W, H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Use default PIL font (no external dependency needed for generate.py)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        font_small = font

    y = 60
    for line in text_lines:
        draw.text((60, y), line, fill=(10, 10, 10), font=font)
        y += 24
        if y > H - 60:
            break

    # Slight rotation
    if abs(rotate_deg) > 0.01:
        img = img.rotate(rotate_deg, expand=False, fillcolor=(255, 255, 255))

    # Add noise
    if noise_level > 0:
        import numpy as np
        arr = np.array(img, dtype=np.int16)
        noise = np.random.randint(-noise_level * 3, noise_level * 3, arr.shape, dtype=np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def generate_scanned_pdf() -> None:
    """Create scanned_policy.pdf: image-only PDF with injected PII."""
    doc_name = "scanned_policy.pdf"
    out_path = OUT_DIR / doc_name
    print(f"\n[1/3] Generating {doc_name}…")

    entities: List[Dict] = []
    gt_text_lines: List[str] = []

    # Fake identities
    author_name = fake.name()
    names = [fake.name() for _ in range(4)]
    emails = [fake.email() for _ in range(3)]
    phones = [fake.phone_number() for _ in range(3)]
    ssn = _fake_ssn()
    emp_id = _fake_employee_id()
    ip = _fake_ip()

    # Fake author in metadata (will be set in PDF metadata, not text layer)
    entities.append(_gt_entity("PERSON", author_name, "metadata/author"))

    # Page content definitions (list of lines per page)
    pages_content: List[List[str]] = []

    # Page 1 — cover / intro
    p1 = [
        "CADENCE FINANCIAL SERVICES",
        "EMPLOYEE DATA PROTECTION POLICY",
        f"Author: {author_name}",
        "Version 3.2  |  Effective Date: 2024-01-15",
        "",
        "This document is CONFIDENTIAL and for internal use only.",
        "Reproduction without authorisation is strictly prohibited.",
        "",
        "Prepared by the Compliance and Risk Management Division.",
        "Questions: compliance@cadencefinancial.com",
    ]
    pages_content.append(p1)
    entities.append(_gt_entity("PERSON", author_name, "page:1"))
    entities.append(_gt_entity("EMAIL", "compliance@cadencefinancial.com", "page:1"))
    gt_text_lines.extend(p1)

    # Page 2 — employee records
    p2 = [
        "SECTION 2: EMPLOYEE RECORDS",
        "",
        f"Employee Name: {names[0]}",
        f"Employee ID: {emp_id}",
        f"SSN: {ssn}",
        f"Email: {emails[0]}",
        f"Phone: {phones[0]}",
        f"IP Address (workstation): {ip}",
        "",
        "Note: The above record is synthetic and for testing only.",
    ]
    pages_content.append(p2)
    entities.append(_gt_entity("PERSON",      names[0],  "page:2"))
    entities.append(_gt_entity("EMPLOYEE_ID", emp_id,    "page:2"))
    entities.append(_gt_entity("SSN",         ssn,       "page:2"))
    entities.append(_gt_entity("EMAIL",       emails[0], "page:2"))
    entities.append(_gt_entity("PHONE",       phones[0], "page:2"))
    entities.append(_gt_entity("IP_ADDRESS",  ip,        "page:2"))
    gt_text_lines.extend(p2)

    # Page 3 — UPPERCASE name (tricky)
    upper_name = names[1].upper()
    p3 = [
        "SECTION 3: AUTHORISED SIGNATORIES",
        "",
        f"Primary Signatory: {upper_name}",
        f"Contact: {emails[1]}",
        f"Mobile: {phones[1]}",
        "",
        "The above individuals are authorised to sign financial agreements.",
        "AcmeCorp Ltd and GlobalTech Solutions are NOT personal names.",
    ]
    pages_content.append(p3)
    entities.append(_gt_entity("PERSON", upper_name, "page:3", tricky=True))
    entities.append(_gt_entity("EMAIL",  emails[1],  "page:3"))
    entities.append(_gt_entity("PHONE",  phones[1],  "page:3"))
    gt_text_lines.extend(p3)

    # Page 4 — ID that looks like a phone (tricky)
    weird_id = "EMP-0987654321"  # 10-digit suffix looks like a phone
    p4 = [
        "SECTION 4: SYSTEM ACCESS CONTROLS",
        "",
        f"Approved By: {names[2]}",
        f"Staff ID: {weird_id}",
        f"Email: {emails[2]}",
        "",
        "Acme Compliance Platform v2.3 (not a person's name).",
        "ProductID: PRD-20240115 (not PII).",
    ]
    pages_content.append(p4)
    entities.append(_gt_entity("PERSON",      names[2],  "page:4"))
    entities.append(_gt_entity("EMPLOYEE_ID", weird_id,  "page:4", tricky=True))
    entities.append(_gt_entity("EMAIL",       emails[2], "page:4"))
    gt_text_lines.extend(p4)

    # Page 5 — failed-Luhn card (must NOT be detected) + invalid Aadhaar
    bad_card = _fake_non_luhn_card()
    bad_aadhaar = _fake_aadhaar_invalid()
    p5 = [
        "SECTION 5: PAYMENT INSTRUMENT NOTES",
        "",
        f"Test card (intentionally invalid): {bad_card}",
        f"Aadhaar (partial/OCR noise): {bad_aadhaar}",
        f"Reviewed by: {names[3]}",
        f"Mobile: {phones[2]}",
        "",
        "WARNING: The card number above is invalid and must not be flagged.",
    ]
    pages_content.append(p5)
    # bad_card intentionally excluded from GT (should NOT be detected)
    # bad_aadhaar: should be flagged at lower confidence due to Verhoeff fail
    entities.append(_gt_entity("AADHAAR", bad_aadhaar, "page:5", tricky=True))
    entities.append(_gt_entity("PERSON",  names[3],    "page:5"))
    entities.append(_gt_entity("PHONE",   phones[2],   "page:5"))
    gt_text_lines.extend(p5)

    # Page 6 — policy footer / signature
    p6 = [
        "SECTION 6: APPROVAL",
        "",
        f"Document Owner: {names[0]}",
        "This policy is approved for use within Cadence Financial Services.",
        "Next review date: 2025-01-15.",
        "",
        "Cadence Financial Services Pty Ltd",
        "ABN 12 345 678 901",
        "NOT a personal identifier — company registration number.",
    ]
    pages_content.append(p6)
    entities.append(_gt_entity("PERSON", names[0], "page:6"))
    gt_text_lines.extend(p6)

    # --- Render pages as images → image-only PDF ---
    # We use PyMuPDF to assemble the images into a PDF (no text layer)
    pdf_doc = fitz.open()
    rotations = [0.0, 0.8, -0.5, 1.2, -0.3, 0.6]
    noises    = [4,   6,    5,    7,    5,    4  ]

    for i, (lines, rot, noise) in enumerate(zip(pages_content, rotations, noises)):
        img = _render_page_image(lines, rotate_deg=rot, noise_level=noise)
        # Convert PIL → bytes
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        img_bytes = buf.read()

        # Insert as an image-only page
        page = pdf_doc.new_page(width=595, height=842)  # A4 points
        rect = fitz.Rect(0, 0, 595, 842)
        page.insert_image(rect, stream=img_bytes)

    # Set metadata (author — a PII vector!)
    pdf_doc.set_metadata({
        "author": author_name,
        "title": "Employee Data Protection Policy",
        "subject": "Confidential HR Policy",
    })
    pdf_doc.save(str(out_path))
    pdf_doc.close()
    print(f"  Written      → {out_path}")

    _save_gt("scanned_policy.json", entities, doc_name)
    _save_gt_text("scanned_policy_text.txt", "\n".join(gt_text_lines))


# ---------------------------------------------------------------------------
# 2. training_guide.docx
# ---------------------------------------------------------------------------

def generate_training_docx() -> None:
    """Create training_guide.docx with headings, tables, header/footer, footnote, image."""
    doc_name = "training_guide.docx"
    out_path = OUT_DIR / doc_name
    print(f"\n[2/3] Generating {doc_name}…")

    entities: List[Dict] = []
    gt_text_lines: List[str] = []

    # Fake data
    author   = fake.name()
    modifier = fake.name()
    employees = [(fake.name(), fake.email(), fake.phone_number(), _fake_employee_id()) for _ in range(5)]
    footer_name = fake.name()
    fn_name  = fake.name()
    fn_email = fake.email()
    ssn2     = _fake_ssn()
    pan      = "ABCDE" + str(random.randint(1000, 9999)) + "F"  # PAN format

    # --- Core properties ---
    doc = Document()
    doc.core_properties.author = author
    doc.core_properties.last_modified_by = modifier
    entities.append(_gt_entity("PERSON", author,   "metadata/author"))
    entities.append(_gt_entity("PERSON", modifier, "metadata/last_modified_by"))

    # --- Header ---
    section = doc.sections[0]
    header = section.header
    hp = header.paragraphs[0]
    hp.text = f"CADENCE FINANCIAL SERVICES — Prepared by: {footer_name}"
    hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    entities.append(_gt_entity("PERSON", footer_name, "header"))
    gt_text_lines.append(f"CADENCE FINANCIAL SERVICES — Prepared by: {footer_name}")

    # --- Footer ---
    footer = section.footer
    fp = footer.paragraphs[0]
    fp.text = f"Confidential | Owner: {footer_name} | Page"
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # footer_name already in GT

    # --- Body ---
    doc.add_heading("Employee Training Guide", 0)
    gt_text_lines.append("Employee Training Guide")

    doc.add_heading("1. Introduction", 1)
    gt_text_lines.append("1. Introduction")

    intro = (
        "This guide provides comprehensive training materials for all Cadence "
        "Financial Services employees. It covers data protection, compliance, "
        "and internal procedures. All information herein is strictly confidential."
    )
    doc.add_paragraph(intro)
    gt_text_lines.append(intro)

    # Add ~15 filler paragraphs
    for i in range(15):
        para_text = fake.paragraph(nb_sentences=4)
        doc.add_paragraph(para_text)
        gt_text_lines.append(para_text)

    doc.add_heading("2. Employee Roster", 1)
    gt_text_lines.append("2. Employee Roster")

    # --- Table 1: Name/Email/Phone/Employee ID (PII table) ---
    headers = ["Name", "Email", "Phone", "Employee ID"]
    table1 = doc.add_table(rows=1, cols=4)
    table1.style = "Table Grid"
    hdr_cells = table1.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h

    for emp_name, emp_email, emp_phone, emp_id in employees:
        row_cells = table1.add_row().cells
        row_cells[0].text = emp_name
        row_cells[1].text = emp_email
        row_cells[2].text = emp_phone
        row_cells[3].text = emp_id
        entities.append(_gt_entity("PERSON",      emp_name,  "table:1"))
        entities.append(_gt_entity("EMAIL",       emp_email, "table:1"))
        entities.append(_gt_entity("PHONE",       emp_phone, "table:1"))
        entities.append(_gt_entity("EMPLOYEE_ID", emp_id,    "table:1"))
        gt_text_lines.append(f"{emp_name} | {emp_email} | {emp_phone} | {emp_id}")

    doc.add_paragraph("")

    # --- Mid-section text ---
    doc.add_heading("3. Compliance Requirements", 1)
    gt_text_lines.append("3. Compliance Requirements")

    for i in range(10):
        doc.add_paragraph(fake.paragraph(nb_sentences=5))

    # --- Tricky: name split across runs (same paragraph, different runs) ---
    split_first = fake.first_name()
    split_last  = fake.last_name()
    split_full  = f"{split_first} {split_last}"
    p = doc.add_paragraph("Approved by: ")
    run1 = p.add_run(split_first)
    run2 = p.add_run(f" {split_last}")
    entities.append(_gt_entity("PERSON", split_full, "body", tricky=True))
    gt_text_lines.append(f"Approved by: {split_full}")

    # --- SSN in body text ---
    doc.add_paragraph(
        f"National Identifier (SSN): {ssn2} — to be stored in encrypted form only."
    )
    entities.append(_gt_entity("SSN", ssn2, "body"))
    gt_text_lines.append(f"National Identifier (SSN): {ssn2} — to be stored in encrypted form only.")

    # --- PAN ---
    doc.add_paragraph(f"Tax Reference (PAN): {pan}")
    entities.append(_gt_entity("PAN", pan, "body"))
    gt_text_lines.append(f"Tax Reference (PAN): {pan}")

    # More filler
    for i in range(15):
        doc.add_paragraph(fake.paragraph(nb_sentences=4))

    doc.add_heading("4. IT Security", 1)
    gt_text_lines.append("4. IT Security")

    # --- Table 2: Non-PII table (false positive trap) ---
    table2 = doc.add_table(rows=4, cols=3)
    table2.style = "Table Grid"
    t2_data = [
        ["System", "Version", "Vendor"],
        ["Acme DataVault", "3.1", "AcmeCorp Ltd"],
        ["SecureStore Pro", "2.0", "GlobalTech Solutions"],
        ["Compliance Hub", "1.5", "Cadence Internal"],
    ]
    for ri, row_data in enumerate(t2_data):
        for ci, val in enumerate(row_data):
            table2.rows[ri].cells[ci].text = val
        gt_text_lines.append(" | ".join(row_data))

    # --- Table 3: Simple key-value ---
    table3 = doc.add_table(rows=3, cols=2)
    table3.style = "Table Grid"
    t3_data = [
        ["Policy Number", "POL-2024-001"],
        ["Review Cycle", "Annual"],
        ["Classification", "Internal Use Only"],
    ]
    for ri, row_data in enumerate(t3_data):
        for ci, val in enumerate(row_data):
            table3.rows[ri].cells[ci].text = val
        gt_text_lines.append(" | ".join(row_data))

    # More filler
    for i in range(15):
        doc.add_paragraph(fake.paragraph(nb_sentences=4))

    # --- Footnote ---
    # python-docx does not have a built-in footnote API; we patch via XML
    from docx.oxml.ns import qn as _qn
    from lxml import etree as ET

    fn_text_content = f"For enquiries contact {fn_name} at {fn_email}."
    entities.append(_gt_entity("PERSON", fn_name,  "footnote:1"))
    entities.append(_gt_entity("EMAIL",  fn_email, "footnote:1"))
    gt_text_lines.append(fn_text_content)

    # Build footnote XML and attach it
    fn_id = "1"
    fn_xml = f"""<w:footnote xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        w:type="normal" w:id="{fn_id}">
    <w:p><w:r><w:t xml:space="preserve">{fn_text_content}</w:t></w:r></w:p>
    </w:footnote>"""
    fn_elem = ET.fromstring(fn_xml)

    # Attach footnote to document part (create footnotes part if missing)
    try:
        fn_part = doc.part.footnotes
        fn_root = ET.fromstring(fn_part._blob)
        fn_root.append(fn_elem)
        fn_part._blob = ET.tostring(fn_root, xml_declaration=True, encoding="UTF-8", standalone=True)
    except Exception:
        pass  # footnote injection skipped if part not available

    # Reference marker in body
    p_fn = doc.add_paragraph("See footnote for contact information")
    run_fn = p_fn.add_run("¹")
    run_fn.font.superscript = True

    # --- Embedded PNG image with PII text ---
    img_emp_name  = fake.name()
    img_emp_email = fake.email()
    img_text_lines = [
        "INTERNAL SCREENSHOT",
        f"Name: {img_emp_name}",
        f"Email: {img_emp_email}",
        "System: HR Portal v4.2",
    ]
    img = Image.new("RGB", (400, 180), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    try:
        fnt = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        fnt = ImageFont.load_default()
    y_offset = 15
    for line in img_text_lines:
        draw.text((10, y_offset), line, fill=(20, 20, 20), font=fnt)
        y_offset += 25

    entities.append(_gt_entity("PERSON", img_emp_name,  "image:1"))
    entities.append(_gt_entity("EMAIL",  img_emp_email, "image:1"))
    gt_text_lines.extend(img_text_lines)

    img_buf = io.BytesIO()
    img.save(img_buf, format="PNG")
    img_buf.seek(0)
    doc.add_paragraph("Attached screenshot (embedded image):")
    doc.add_picture(img_buf, width=Inches(4))

    doc.save(str(out_path))
    print(f"  Written      → {out_path}")

    _save_gt("training_guide.json", entities, doc_name)
    _save_gt_text("training_guide_text.txt", "\n".join(gt_text_lines))


# ---------------------------------------------------------------------------
# 3. org_pack.pptx
# ---------------------------------------------------------------------------

def _add_text_box(slide, left, top, width, height, text, font_size=14, bold=False):
    """Add a text box to a PPTX slide."""
    txBox = slide.shapes.add_textbox(
        PInches(left), PInches(top), PInches(width), PInches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = PPt(font_size)
    run.font.bold = bold
    return txBox


def generate_org_pptx() -> None:
    """Create org_pack.pptx: directory table, RACI table, org chart, picture, notes."""
    doc_name = "org_pack.pptx"
    out_path = OUT_DIR / doc_name
    print(f"\n[3/3] Generating {doc_name}…")

    entities: List[Dict] = []
    gt_text_lines: List[str] = []

    prs = Presentation()
    prs.slide_width  = PInches(10)
    prs.slide_height = PInches(7.5)

    # Core properties
    author_p  = fake.name()
    modifier_p = fake.name()
    prs.core_properties.author = author_p
    prs.core_properties.last_modified_by = modifier_p
    entities.append(_gt_entity("PERSON", author_p,   "metadata/author"))
    entities.append(_gt_entity("PERSON", modifier_p, "metadata/last_modified_by"))

    blank_layout = prs.slide_layouts[6]  # blank

    # --- Slide 1: Title ---
    slide1 = prs.slides.add_slide(prs.slide_layouts[0])
    slide1.shapes.title.text = "Cadence Financial Services"
    slide1.placeholders[1].text = "Organisational Pack — Confidential"
    gt_text_lines.append("Cadence Financial Services")
    gt_text_lines.append("Organisational Pack — Confidential")

    # --- Slide 2: Role-holder directory table ---
    dir_names   = [fake.name() for _ in range(5)]
    dir_titles  = ["Chief Executive Officer", "Chief Risk Officer",
                   "Head of Compliance", "VP Engineering", "HR Director"]
    dir_emails  = [fake.email() for _ in range(5)]
    dir_phones  = [fake.phone_number() for _ in range(5)]

    slide2 = prs.slides.add_slide(blank_layout)
    _add_text_box(slide2, 0.3, 0.2, 9, 0.5, "Directory of Role Holders", font_size=20, bold=True)
    gt_text_lines.append("Directory of Role Holders")

    tbl_cols = 4
    tbl_rows = 6  # header + 5 people
    left2 = PInches(0.3); top2 = PInches(0.9); w2 = PInches(9.4); h2 = PInches(3.5)
    table2 = slide2.shapes.add_table(tbl_rows, tbl_cols, left2, top2, w2, h2).table
    for ci, hdr in enumerate(["Name", "Title", "Email", "Phone"]):
        table2.cell(0, ci).text = hdr
    for ri, (n, t, e, ph) in enumerate(zip(dir_names, dir_titles, dir_emails, dir_phones)):
        table2.cell(ri + 1, 0).text = n
        table2.cell(ri + 1, 1).text = t
        table2.cell(ri + 1, 2).text = e
        table2.cell(ri + 1, 3).text = ph
        loc = "slide:2/table:1"
        entities.append(_gt_entity("PERSON", n,  loc))
        entities.append(_gt_entity("EMAIL",  e,  loc))
        entities.append(_gt_entity("PHONE",  ph, loc))
        gt_text_lines.append(f"{n} | {t} | {e} | {ph}")

    # --- Slide 3: RACI table ---
    raci_names = [fake.name() for _ in range(4)]
    slide3 = prs.slides.add_slide(blank_layout)
    _add_text_box(slide3, 0.3, 0.2, 9, 0.5, "RACI Matrix", font_size=20, bold=True)
    gt_text_lines.append("RACI Matrix")

    raci_headers = ["Task", "Responsible", "Accountable", "Consulted", "Informed"]
    raci_tasks   = ["Risk Assessment", "Policy Review", "Audit Report", "System Access"]
    raci_tbl = slide3.shapes.add_table(
        5, 5, PInches(0.3), PInches(0.9), PInches(9.4), PInches(3.5)).table
    for ci, h in enumerate(raci_headers):
        raci_tbl.cell(0, ci).text = h
    for ri, task in enumerate(raci_tasks):
        raci_tbl.cell(ri + 1, 0).text = task
        # Assign names (cyclic)
        for ci in range(1, 5):
            name = raci_names[(ri + ci) % len(raci_names)]
            raci_tbl.cell(ri + 1, ci).text = name
            entities.append(_gt_entity("PERSON", name, "slide:3/table:1"))
        gt_text_lines.append(f"{task} | {' | '.join(raci_names[(ri + ci) % len(raci_names)] for ci in range(1,5))}")

    # --- Slide 4: Org chart (grouped text boxes simulating shapes) ---
    slide4 = prs.slides.add_slide(blank_layout)
    _add_text_box(slide4, 0.3, 0.1, 9, 0.4, "Organisation Chart", font_size=20, bold=True)
    gt_text_lines.append("Organisation Chart")

    org_people = [(fake.name(), title) for title in
                  ["CEO", "CRO", "CFO", "Head of IT", "Head of HR"]]
    for i, (name, title) in enumerate(org_people):
        col = i % 3
        row = i // 3
        _add_text_box(slide4, 0.5 + col * 3.2, 0.8 + row * 2.0, 2.8, 0.8,
                      f"{name}\n{title}", font_size=12)
        entities.append(_gt_entity("PERSON", name, "slide:4"))
        gt_text_lines.append(f"{name}\n{title}")

    # --- Slide 5: Policy text ---
    slide5 = prs.slides.add_slide(blank_layout)
    _add_text_box(slide5, 0.3, 0.2, 9, 0.4, "Data Protection Policy", font_size=20, bold=True)
    policy_body = (
        "All employees must comply with the data protection framework. "
        "Violations are subject to disciplinary action. "
        "Refer to the Employee Handbook for full details."
    )
    _add_text_box(slide5, 0.3, 0.9, 9.4, 3, policy_body)
    gt_text_lines.append("Data Protection Policy")
    gt_text_lines.append(policy_body)

    # --- Slide 6: Picture with embedded PII text ---
    img_name  = fake.name()
    img_email = fake.email()
    img_lines = [f"User: {img_name}", f"Email: {img_email}", "Access Level: Admin"]
    img = Image.new("RGB", (500, 160), color=(230, 240, 250))
    draw = ImageDraw.Draw(img)
    try:
        fnt = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        fnt = ImageFont.load_default()
    y_off = 15
    for line in img_lines:
        draw.text((15, y_off), line, fill=(20, 20, 60), font=fnt)
        y_off += 30

    slide6 = prs.slides.add_slide(blank_layout)
    _add_text_box(slide6, 0.3, 0.1, 9, 0.4, "System Access Screenshot", font_size=20, bold=True)
    img_buf = io.BytesIO()
    img.save(img_buf, format="PNG")
    img_buf.seek(0)
    slide6.shapes.add_picture(img_buf, PInches(1), PInches(0.8), PInches(8), PInches(2))
    entities.append(_gt_entity("PERSON", img_name,  "slide:6/image"))
    entities.append(_gt_entity("EMAIL",  img_email, "slide:6/image"))
    gt_text_lines.extend(img_lines)

    # --- Slide 7: Compliance contacts ---
    comp_name  = fake.name()
    comp_email = fake.email()
    comp_phone = fake.phone_number()
    slide7 = prs.slides.add_slide(blank_layout)
    _add_text_box(slide7, 0.3, 0.2, 9, 0.4, "Compliance Contacts", font_size=20, bold=True)
    _add_text_box(slide7, 0.3, 0.9, 9.4, 2,
                  f"Owner: {comp_name}\nEmail: {comp_email}\nPhone: {comp_phone}")
    entities.append(_gt_entity("PERSON", comp_name,  "slide:7"))
    entities.append(_gt_entity("EMAIL",  comp_email, "slide:7"))
    entities.append(_gt_entity("PHONE",  comp_phone, "slide:7"))
    gt_text_lines.extend([f"Owner: {comp_name}", f"Email: {comp_email}", f"Phone: {comp_phone}"])

    # Speaker notes with PII
    notes_name  = fake.name()
    notes_email = fake.email()
    notes_tf = slide7.notes_slide.notes_text_frame
    notes_text = (
        f"Presenter: {notes_name} ({notes_email}). "
        "Remind attendees that all data is confidential."
    )
    notes_tf.text = notes_text
    entities.append(_gt_entity("PERSON", notes_name,  "slide:7/notes"))
    entities.append(_gt_entity("EMAIL",  notes_email, "slide:7/notes"))
    gt_text_lines.append(notes_text)

    # --- Slide 8: Summary ---
    slide8 = prs.slides.add_slide(blank_layout)
    _add_text_box(slide8, 0.3, 0.2, 9, 0.4, "Summary", font_size=20, bold=True)
    _add_text_box(slide8, 0.3, 0.9, 9.4, 3,
                  "This pack is confidential. All names, contact details, and identifiers "
                  "contained herein are strictly for internal use. "
                  "Please contact compliance@cadencefinancial.com with any questions.")
    entities.append(_gt_entity("EMAIL", "compliance@cadencefinancial.com", "slide:8"))
    gt_text_lines.append("Summary")
    gt_text_lines.append("compliance@cadencefinancial.com")

    prs.save(str(out_path))
    print(f"  Written      → {out_path}")

    _save_gt("org_pack.json", entities, doc_name)
    _save_gt_text("org_pack_text.txt", "\n".join(gt_text_lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating synthetic documents…")
    generate_scanned_pdf()
    generate_training_docx()
    generate_org_pptx()
    print("\n✅ All synthetic documents generated.")
    print(f"   Documents : {OUT_DIR}")
    print(f"   GT files  : {GT_DIR}")
