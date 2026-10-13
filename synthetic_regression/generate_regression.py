"""
synthetic_regression/generate_regression.py
--------------------------------------------
Generates the regression pack under /synthetic_regression reproducing
the failure patterns a) through p) from the independent review:
- scanned_policy.pdf: image-only pages at 1654 x 2339 px (~200 DPI), no text layer,
  with slight skew/noise, containing figures k, l, m, n, p.
- training_guide.docx: containing dark admin console screenshot at 600, 1500, 2500 px,
  SVG vector image with text, tables, headers, and narrative prose.
- org_pack.pptx: presentation with dark admin console, dashboard, RACI directory.
- ground_truth/*.json: ground truth for every injected entity with source tags:
  native | page_ocr | region_ocr | svg | emf, plus must_not_redact phrases
  and expected figure tokens.

Deterministic seed = 42. Invented data only.
"""
from __future__ import annotations

import io
import json
import os
import random
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from pptx import Presentation
from pptx.util import Inches as PInches, Pt as PPt

random.seed(42)
np.random.seed(42)

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / "synthetic_regression"
GT_DIR = OUT_DIR / "ground_truth"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Font helper
# ---------------------------------------------------------------------------
def _get_font(size: int = 24) -> ImageFont.ImageFont:
    for font_name in ["arial.ttf", "calibri.ttf", "segoeui.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Drawing Figures (k, l, m, n, p)
# ---------------------------------------------------------------------------
def create_form_and_badge_image() -> Tuple[Image.Image, List[str]]:
    """
    Figure (k): Scanned form table (mixing DOB, SSN, PAN, Passport in last column)
    plus ID-badge image with a DOB line. Skewed ~3 degrees.
    """
    W, H = 1200, 750
    img = Image.new("RGB", (W, H), color=(250, 250, 248))
    draw = ImageDraw.Draw(img)
    font = _get_font(22)
    font_bold = _get_font(24)
    font_small = _get_font(18)

    # Form title
    draw.text((40, 30), "CADENCE AUDIT VERIFICATION FORM (FORM-882)", fill=(20, 20, 20), font=font_bold)
    draw.rectangle([40, 70, 750, 480], outline=(40, 40, 40), width=2)
    draw.line([40, 120, 750, 120], fill=(40, 40, 40), width=2)
    draw.line([250, 70, 250, 480], fill=(40, 40, 40), width=2)
    draw.line([480, 70, 480, 480], fill=(40, 40, 40), width=2)

    # Table headers
    draw.text((60, 85), "Employee Name", fill=(30, 30, 30), font=font_bold)
    draw.text((270, 85), "Work Email", fill=(30, 30, 30), font=font_bold)
    draw.text((500, 85), "Identification / DOB", fill=(30, 30, 30), font=font_bold)

    rows = [
        ("Harold V. Finch", "h.finch@cadencecorp.net", "DOB 14 July 1991"),
        ("Marcus Vance", "mvance@cadencecorp.net", "SSN 900-55-2043"),
        ("Sunita Narayan", "snarayan@cadencecorp.net", "PAN ABCDE1234F"),
        ("Elena Rostova", "erostova@cadencecorp.net", "Passport N8841295"),
    ]

    expected_tokens = [
        "Harold", "Finch", "h.finch@cadencecorp.net", "14", "July", "1991",
        "Marcus", "Vance", "mvance@cadencecorp.net", "900-55-2043",
        "Sunita", "Narayan", "snarayan@cadencecorp.net", "ABCDE1234F",
        "Elena", "Rostova", "erostova@cadencecorp.net", "N8841295"
    ]

    y = 140
    for name, email, ident in rows:
        draw.text((50, y), name, fill=(20, 20, 20), font=font)
        draw.text((260, y), email, fill=(20, 20, 20), font=font)
        draw.text((490, y), ident, fill=(20, 20, 20), font=font)
        draw.line([40, y + 45, 750, y + 45], fill=(180, 180, 180), width=1)
        y += 60

    # ID Badge Block
    draw.rectangle([800, 70, 1160, 480], outline=(0, 60, 130), width=3, fill=(245, 248, 255))
    draw.rectangle([800, 70, 1160, 130], fill=(0, 60, 130))
    draw.text((840, 85), "CADENCE ACCESS BADGE", fill=(255, 255, 255), font=font_bold)
    draw.rectangle([830, 150, 930, 260], outline=(100, 100, 100), width=1, fill=(220, 225, 230))
    draw.text((845, 195), "[PHOTO]", fill=(80, 80, 80), font=font_small)

    badge_tokens = ["SECURITY", "BADGE", "Theresa", "Lindqvist", "DIR-0042", "DATE", "OF", "BIRTH", "28", "February", "1979"]
    expected_tokens.extend(badge_tokens)

    draw.text((950, 160), "Theresa Lindqvist", fill=(20, 20, 20), font=font_bold)
    draw.text((950, 195), "Role: Director", fill=(50, 50, 50), font=font)
    draw.text((950, 230), "ID: DIR-0042", fill=(50, 50, 50), font=font)

    draw.line([820, 280, 1140, 280], fill=(0, 60, 130), width=1)
    draw.text((830, 300), "DATE OF BIRTH", fill=(80, 80, 80), font=font_bold)
    draw.text((830, 335), "28 February 1979", fill=(10, 10, 10), font=font_bold)
    draw.text((830, 385), "Badge: BG-88914-EXP", fill=(90, 90, 90), font=font_small)

    # Skew ~3 degrees
    skewed = img.rotate(3.0, expand=True, fillcolor=(255, 255, 255), resample=Image.BICUBIC)
    return skewed, expected_tokens


def create_admin_console_image(scale_w: int = 1400) -> Tuple[Image.Image, List[str]]:
    """
    Figure (l): Dark-theme admin console screenshot.
    Dark sidebar, light-grey 11-13px text at page scale, avatar circles,
    10 invented users with names, emails, roles, last-login.
    """
    base_w, base_h = 1400, 800
    img = Image.new("RGB", (base_w, base_h), color=(30, 32, 40))
    draw = ImageDraw.Draw(img)

    sidebar_w = 260
    draw.rectangle([0, 0, sidebar_w, base_h], fill=(22, 24, 30))
    font_title = _get_font(22)
    font_header = _get_font(18)
    font_text = _get_font(16)
    font_sub = _get_font(13)

    # Sidebar items
    draw.text((30, 35), "Cadence Cloud", fill=(220, 220, 230), font=font_title)
    draw.text((30, 90), "Dashboard", fill=(150, 155, 170), font=font_text)
    draw.text((30, 130), "User Management", fill=(80, 160, 255), font=font_text)
    draw.text((30, 170), "Audit Logs", fill=(150, 155, 170), font=font_text)
    draw.text((30, 210), "Security Settings", fill=(150, 155, 170), font=font_text)

    # Top header bar
    draw.rectangle([sidebar_w, 0, base_w, 65], fill=(26, 28, 35))
    draw.text((sidebar_w + 30, 20), "Active Directory Users & IAM Console", fill=(240, 240, 245), font=font_title)

    # Table columns
    col_x = [sidebar_w + 30, sidebar_w + 270, sidebar_w + 580, sidebar_w + 800, sidebar_w + 980]
    draw.text((col_x[0], 85), "User", fill=(160, 165, 180), font=font_header)
    draw.text((col_x[1], 85), "Email Address", fill=(160, 165, 180), font=font_header)
    draw.text((col_x[2], 85), "Role / Group", fill=(160, 165, 180), font=font_header)
    draw.text((col_x[3], 85), "Last Login", fill=(160, 165, 180), font=font_header)
    draw.text((col_x[4], 85), "Status", fill=(160, 165, 180), font=font_header)
    draw.line([sidebar_w + 20, 115, base_w - 20, 115], fill=(50, 54, 66), width=1)

    users = [
        ("M. T. Achebe", "m.achebe@cadencecorp.net", "Security Officer", "2026-04-12 09:14", "Active"),
        ("T. Lindqvist", "t.lindqvist@cadencecorp.net", "Operations Director", "2026-04-12 08:33", "Active"),
        ("P. S. Raghavan", "p.raghavan@cadencecorp.net", "Lead Auditor", "2026-04-11 17:45", "Active"),
        ("Kavita Reddy", "k.reddy@cadencecorp.net", "Compliance Analyst", "2026-04-11 16:20", "Active"),
        ("Arthur Pendelton", "a.pendelton@cadencecorp.net", "Database Admin", "2026-04-11 14:10", "Active"),
        ("Beatrix Thorne", "b.thorne@cadencecorp.net", "Risk Manager", "2026-04-10 11:05", "Active"),
        ("Gideon Cross", "g.cross@cadencecorp.net", "Cloud Architect", "2026-04-10 09:50", "Active"),
        ("Yuki Tanaka", "y.tanaka@cadencecorp.net", "Systems Engineer", "2026-04-09 18:12", "Active"),
        ("Fiona MacLeod", "f.macleod@cadencecorp.net", "Privacy Officer", "2026-04-09 15:40", "Active"),
        ("Damian O'Connor", "d.oconnor@cadencecorp.net", "DevOps Engineer", "2026-04-08 12:25", "Active"),
    ]

    expected_tokens = []
    y = 135
    for name, email, role, last_login, status in users:
        # Avatar circle
        draw.ellipse([col_x[0], y + 2, col_x[0] + 24, y + 26], fill=(55, 95, 160))
        draw.text((col_x[0] + 35, y + 4), name, fill=(235, 238, 245), font=font_text)
        draw.text((col_x[1], y + 4), email, fill=(190, 195, 205), font=font_text)
        draw.text((col_x[2], y + 4), role, fill=(170, 175, 185), font=font_text)
        draw.text((col_x[3], y + 4), last_login, fill=(160, 165, 175), font=font_sub)
        draw.text((col_x[4], y + 4), status, fill=(100, 210, 130), font=font_sub)
        draw.line([sidebar_w + 20, y + 40, base_w - 20, y + 40], fill=(40, 44, 55), width=1)

        expected_tokens.extend([part for part in name.split()])
        expected_tokens.append(email)
        y += 50

    if scale_w != base_w:
        ratio = scale_w / float(base_w)
        scaled_h = int(base_h * ratio)
        img = img.resize((scale_w, scaled_h), Image.LANCZOS)

    return img, expected_tokens


def create_dashboard_image() -> Tuple[Image.Image, List[str]]:
    """
    Figure (m): Dashboard-style figure with bars, percentages,
    and a 'metric owner' column with invented names.
    """
    W, H = 1000, 500
    img = Image.new("RGB", (W, H), color=(252, 252, 254))
    draw = ImageDraw.Draw(img)

    font_title = _get_font(24)
    font_bold = _get_font(20)
    font = _get_font(18)
    font_small = _get_font(15)

    draw.text((40, 30), "KPI DASHBOARD: PRIVACY & RISK METRICS (Q1 2026)", fill=(15, 23, 42), font=font_title)
    draw.rectangle([40, 80, 960, 450], outline=(203, 213, 225), width=2, fill=(255, 255, 255))

    headers = ["Metric Name", "Completion", "Score", "Metric Owner"]
    draw.text((70, 100), headers[0], fill=(71, 85, 105), font=font_bold)
    draw.text((360, 100), headers[1], fill=(71, 85, 105), font=font_bold)
    draw.text((580, 100), headers[2], fill=(71, 85, 105), font=font_bold)
    draw.text((720, 100), headers[3], fill=(71, 85, 105), font=font_bold)
    draw.line([40, 135, 960, 135], fill=(226, 232, 240), width=2)

    rows = [
        ("Access Audit Compliance", 0.94, "94.2%", "Cedric Vance"),
        ("DPIA Verification Velocity", 0.88, "88.5%", "Amina Qasim"),
        ("Vendor SOC2 Intake", 0.76, "76.0%", "P. S. Raghavan"),
        ("Incident Response Testing", 0.98, "98.1%", "Nicolette Dupuis"),
    ]

    expected_tokens = []
    y = 160
    for title, frac, pct_str, owner in rows:
        draw.text((70, y + 5), title, fill=(30, 41, 59), font=font)
        # Progress bar
        draw.rectangle([360, y + 8, 540, y + 26], fill=(241, 245, 249), outline=(203, 213, 225))
        draw.rectangle([360, y + 8, 360 + int(180 * frac), y + 26], fill=(37, 99, 235))
        draw.text((580, y + 5), pct_str, fill=(30, 41, 59), font=font)
        draw.text((720, y + 5), owner, fill=(15, 23, 42), font=font_bold)
        draw.line([40, y + 50, 960, y + 50], fill=(241, 245, 249), width=1)

        expected_tokens.extend([part for part in owner.split()])
        expected_tokens.append(pct_str)
        y += 65

    return img, expected_tokens


def create_identity_record_image() -> Tuple[Image.Image, List[str]]:
    """
    Figure (n): Identity-record style screenshot (~400 px wide)
    Name, work email, mobile number, employee ID, hire date.
    """
    W, H = 420, 320
    img = Image.new("RGB", (W, H), color=(248, 250, 252))
    draw = ImageDraw.Draw(img)

    font_title = _get_font(18)
    font_bold = _get_font(15)
    font = _get_font(15)

    draw.rectangle([10, 10, W - 10, H - 10], outline=(203, 213, 225), width=2, fill=(255, 255, 255))
    draw.rectangle([10, 10, W - 10, 50], fill=(241, 245, 249))
    draw.text((25, 20), "HR Portal: Identity Summary", fill=(30, 41, 59), font=font_title)

    items = [
        ("Employee:", "Lukas Brauer"),
        ("Work Email:", "l.brauer@cadencecorp.net"),
        ("Mobile:", "+49 69 5555 0177"),
        ("Staff ID:", "EMP-38104"),
        ("Hire Date:", "12 March 2021"),
    ]

    expected_tokens = ["Lukas", "Brauer", "l.brauer@cadencecorp.net", "+49 69 5555 0177", "EMP-38104"]
    y = 70
    for label, val in items:
        draw.text((25, y), label, fill=(100, 116, 139), font=font_bold)
        draw.text((150, y), val, fill=(15, 23, 42), font=font)
        y += 42

    return img, expected_tokens


def create_tiny_text_image() -> Tuple[Image.Image, List[str]]:
    """
    Figure (p): Deliberately too small to read at page scale (text height under 7 px)
    with an invented name in it, to test that it is reported as UNVERIFIED.
    """
    W, H = 300, 100
    img = Image.new("RGB", (W, H), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Tiny font: 5 px height
    try:
        font_tiny = ImageFont.truetype("arial.ttf", 6)
    except Exception:
        font_tiny = ImageFont.load_default()

    draw.text((10, 20), "Confidential microprint label:", fill=(120, 120, 120), font=font_tiny)
    draw.text((10, 45), "Custodian: Lysander Vance (Under-resolution)", fill=(120, 120, 120), font=font_tiny)
    expected_tokens = ["Lysander", "Vance"]
    return img, expected_tokens


# ---------------------------------------------------------------------------
# SVG generator for Figure (o)
# ---------------------------------------------------------------------------
def create_svg_content() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 200" width="500" height="200">
  <rect width="100%" height="100%" fill="#f0f4f8"/>
  <rect x="20" y="20" width="460" height="160" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="2"/>
  <text x="40" y="60" font-family="Arial, sans-serif" font-size="18" font-weight="bold" fill="#1e293b">Enterprise Architecture Contact</text>
  <text x="40" y="100" font-family="Arial, sans-serif" font-size="15" fill="#334155">Chief Architect: Zachary Sterling</text>
  <text x="40" y="135" font-family="Arial, sans-serif" font-size="14" fill="#2563eb">z.sterling@cadencecorp.net</text>
</svg>"""


# ---------------------------------------------------------------------------
# Generator: 1. scanned_policy.pdf
# ---------------------------------------------------------------------------
def generate_scanned_pdf() -> Tuple[List[Dict], List[str], Dict]:
    pdf_path = OUT_DIR / "scanned_policy.pdf"
    doc = fitz.open()

    PAGE_W, PAGE_H = 1654, 2339  # ~200 DPI A4
    font_h1 = _get_font(42)
    font_h2 = _get_font(32)
    font_body = _get_font(26)
    font_bold = _get_font(26)

    gt_entities = []
    must_not_redact = []
    figures_info = {}

    # Must-not-redact phrases
    phrases_to_protect = [
        "employee identifiers",
        "national identifiers",
        "distinguishing",
        "contingency",
        "Board Risk Committee",
        "Data Privacy Office",
        "NATIONAL ID",
        "that", "what", "at least", "great", "at an appropriate level",
        "Cadence Financial Group, 100 Wall Street, New York, NY 10005"
    ]
    must_not_redact.extend(phrases_to_protect)

    # PAGE 1: Cover Page with Corporate Address, Header phrases, Phone test block
    img1 = Image.new("RGB", (PAGE_W, PAGE_H), color=(255, 255, 255))
    draw1 = ImageDraw.Draw(img1)

    draw1.text((120, 140), "CADENCE FINANCIAL GROUP", fill=(10, 25, 47), font=font_h1)
    draw1.text((120, 200), "GLOBAL DATA GOVERNANCE & PRIVACY STANDARD", fill=(30, 41, 59), font=font_h2)
    draw1.line([120, 250, PAGE_W - 120, 250], fill=(15, 23, 42), width=4)

    # Corporate address (must NOT be redacted)
    draw1.text((120, 290), "Corporate HQ: Cadence Financial Group, 100 Wall Street, New York, NY 10005", fill=(71, 85, 105), font=font_body)
    draw1.text((120, 340), "Custodian: Board Risk Committee | Supervisory: Data Privacy Office", fill=(71, 85, 105), font=font_body)
    draw1.text((120, 390), "Document ID: POL-2026-PRIV  |  Version 5.4.1", fill=(71, 85, 105), font=font_body)

    # Category a: International phones
    draw1.text((120, 480), "SECTION 1: REGIONAL INCIDENT RESPONSE LINES", fill=(15, 23, 42), font=font_h2)
    draw1.line([120, 525, 800, 525], fill=(71, 85, 105), width=2)

    phones_p1 = [
        ("UK Headquarters (London):", "+44 20 7555 0171", "PHONE"),
        ("UK Direct Dial Desk:", "+44 20 75550171", "PHONE"),
        ("UK Urgent Incident Line:", "+44 7700 555 0163 (mobile)", "PHONE"),
        ("India Hub (Bangalore):", "+91 80 4555 0182", "PHONE"),
        ("India Support (Mumbai):", "+91 22 4555 0117", "PHONE"),
        ("India Emergency Lead:", "+91 98860 55107", "PHONE"),
        ("France Incident Desk:", "+33 1 5555 0144", "PHONE"),
        ("France Direct Line:", "+33 15555 0144", "PHONE"),
        ("Singapore Regional Hub:", "+65 6555 0172", "PHONE"),
        ("Germany Response Center:", "+49 69 5555 0177", "PHONE"),
        ("Sweden Operations Desk:", "+46 8 5555 0121", "PHONE"),
        ("Sweden Incident On-call:", "+46 70 555 0188", "PHONE"),
        ("Poland Operations Center:", "+48 22 555 0164", "PHONE"),
        ("Poland Emergency Hotline:", "+48 512 555 0180", "PHONE"),
        ("Philippines Support Desk:", "+63 917 555 0119", "PHONE"),
        ("US Operations Direct:", "+1 (646) 555-0203", "PHONE"),
        ("US Toll Desk:", "+1 212 555-0147", "PHONE"),
    ]

    y = 560
    for label, num, etype in phones_p1:
        draw1.text((120, y), label, fill=(50, 60, 70), font=font_body)
        draw1.text((580, y), num, fill=(10, 10, 10), font=font_bold)
        # In ground truth, the phone number is the target (suffix '(mobile)' not included in phone span)
        clean_num = num.replace(" (mobile)", "")
        gt_entities.append({
            "type": etype,
            "text": clean_num,
            "location": "page:1",
            "source": "page_ocr",
            "category": "a",
        })
        y += 50

    # Category h: Ordinary English sentences containing 'that', 'what', 'at least', 'great', 'at an appropriate level'
    y += 40
    draw1.text((120, y), "Operational Guidelines & Thresholds:", fill=(15, 23, 42), font=font_h2)
    y += 50
    non_email_lines = [
        "Ensure that all technical safeguards are maintained at an appropriate level.",
        "Consider what controls are needed so that risk-taking is deliberate and monitored.",
        "The team did a great job reviewing distinguishing characteristics of the contingency plan.",
        "Maintain at least three backup redundant paths across all operational clusters."
    ]
    for line in non_email_lines:
        draw1.text((120, y), line, fill=(30, 41, 59), font=font_body)
        y += 45

    # Category j: Date range plus phone fragment resembling 12-digit number
    y += 20
    draw1.text((120, y), "Policy Approval History: Drafted 21 Apr 2026 +44 20 7555 0171", fill=(30, 41, 59), font=font_body)

    # Slight rotation (0.2 degrees) + noise
    arr1 = np.array(img1, dtype=np.int16)
    noise1 = np.random.randint(-4, 5, arr1.shape, dtype=np.int16)
    img1 = Image.fromarray(np.clip(arr1 + noise1, 0, 255).astype(np.uint8))
    img1 = img1.rotate(0.2, expand=False, fillcolor=(255, 255, 255), resample=Image.BICUBIC)

    # PAGE 2: Narrative prose, 9xx SSNs, addresses, card/account fragments, IDs, Figure (k)
    img2 = Image.new("RGB", (PAGE_W, PAGE_H), color=(255, 255, 255))
    draw2 = ImageDraw.Draw(img2)

    draw2.text((120, 120), "SECTION 2: COMPLIANCE INVESTIGATION RECORDS", fill=(10, 25, 47), font=font_h1)
    draw2.line([120, 180, PAGE_W - 120, 180], fill=(15, 23, 42), width=3)

    # Category b & e: Narrative prose with 9xx SSN and last four, account fragments
    prose_lines = [
        ("The investigation file records participant SSN 900-55-2043 during compliance intake.", "900-55-2043", "SSN", "b"),
        ("A cross-reference audit was initiated showing SSN 900-42-7781 in the secondary register.", "900-42-7781", "SSN", "b"),
        ("Please confirm the last four digits of your social security number (8827) before access.", "8827", "SSN", "b"),
        ("The client reported a compromised corporate card ending 2218 on Monday.", "card ending 2218", "ACCOUNT_NUMBER", "e"),
        ("Wire confirmation sent for savings account ending 7741 yesterday afternoon.", "savings account ending 7741", "ACCOUNT_NUMBER", "e"),
    ]

    y = 220
    for full_line, target, etype, cat in prose_lines:
        draw2.text((120, y), full_line, fill=(30, 41, 59), font=font_body)
        gt_entities.append({
            "type": etype,
            "text": target,
            "location": "page:2",
            "source": "page_ocr",
            "category": cat,
        })
        y += 50

    # Category d: Home addresses
    y += 20
    draw2.text((120, y), "Participant Residential Declarations:", fill=(15, 23, 42), font=font_h2)
    y += 50
    address_lines = [
        ("Official filing: home address 88 Willowbank Court, Hoboken, NJ 07030 on record.", "88 Willowbank Court, Hoboken, NJ 07030", "d"),
        ("The employee updated his home address, 1180 Ashwood Lane, Apartment 4C, Jersey City, NJ 07302 last month.", "1180 Ashwood Lane, Apartment 4C, Jersey City, NJ 07302", "d"),
        ("Notice of audit dispatched to your home at 2214 Palmetto Crescent, Coral Gables, FL 33134 via courier.", "2214 Palmetto Crescent, Coral Gables, FL 33134", "d"),
    ]
    for full_line, addr, cat in address_lines:
        draw2.text((120, y), full_line, fill=(30, 41, 59), font=font_body)
        gt_entities.append({
            "type": "ADDRESS",
            "text": addr,
            "location": "page:2",
            "source": "page_ocr",
            "category": cat,
        })
        y += 50

    # Category f: Other identifiers
    y += 20
    draw2.text((120, y), "Foreign & Regulatory Identity Mappings:", fill=(15, 23, 42), font=font_h2)
    y += 50
    other_ids = [
        ("Polish resident verification matched PESEL 890412xxxxx in the registry.", "890412xxxxx", "PESEL", "f"),
        ("Authorisation granted by DIR-0042 for emergency perimeter access.", "DIR-0042", "DIRECTOR_ID", "f"),
        ("Vendor contracting completed under MER-IN-4471 and verified.", "MER-IN-4471", "VENDOR_ID", "f"),
        ("European regional billing associated with MER-PL-1187 and MER-PH-2214.", "MER-PL-1187", "VENDOR_ID", "f"),
        ("Staff record indexed under EMP-38104 in directory.", "EMP-38104", "EMPLOYEE_ID", "f"),
    ]
    gt_entities.append({
        "type": "VENDOR_ID",
        "text": "MER-PH-2214",
        "location": "page:2",
        "source": "page_ocr",
        "category": "f",
    })
    for full_line, target, etype, cat in other_ids:
        draw2.text((120, y), full_line, fill=(30, 41, 59), font=font_body)
        gt_entities.append({
            "type": etype,
            "text": target,
            "location": "page:2",
            "source": "page_ocr",
            "category": cat,
        })
        y += 50

    # Embed Figure (k): Scanned Form & Badge Image inside Page 2
    form_img, form_tokens = create_form_and_badge_image()
    # Paste form image at y=1100
    img2.paste(form_img, (120, y + 30))
    figures_info["figure_k"] = {
        "page": 2,
        "type": "form_and_badge",
        "tokens": form_tokens,
        "bbox": [120, y + 30, 120 + form_img.width, y + 30 + form_img.height],
    }

    # Add Figure (k) PII entities to ground truth with source='region_ocr'
    gt_entities.extend([
        {"type": "PERSON", "text": "Harold V. Finch", "location": "page:2", "source": "region_ocr", "category": "k"},
        {"type": "EMAIL", "text": "h.finch@cadencecorp.net", "location": "page:2", "source": "region_ocr", "category": "k"},
        {"type": "DATE_OF_BIRTH", "text": "14 July 1991", "location": "page:2", "source": "region_ocr", "category": "k"},
        {"type": "PERSON", "text": "Marcus Vance", "location": "page:2", "source": "region_ocr", "category": "k"},
        {"type": "EMAIL", "text": "mvance@cadencecorp.net", "location": "page:2", "source": "region_ocr", "category": "k"},
        {"type": "SSN", "text": "900-55-2043", "location": "page:2", "source": "region_ocr", "category": "k"},
        {"type": "PERSON", "text": "Sunita Narayan", "location": "page:2", "source": "region_ocr", "category": "k"},
        {"type": "EMAIL", "text": "snarayan@cadencecorp.net", "location": "page:2", "source": "region_ocr", "category": "k"},
        {"type": "PAN", "text": "ABCDE1234F", "location": "page:2", "source": "region_ocr", "category": "k"},
        {"type": "PERSON", "text": "Elena Rostova", "location": "page:2", "source": "region_ocr", "category": "k"},
        {"type": "EMAIL", "text": "erostova@cadencecorp.net", "location": "page:2", "source": "region_ocr", "category": "k"},
        {"type": "PASSPORT", "text": "N8841295", "location": "page:2", "source": "region_ocr", "category": "k"},
        {"type": "PERSON", "text": "Theresa Lindqvist", "location": "page:2", "source": "region_ocr", "category": "k"},
        {"type": "DIRECTOR_ID", "text": "DIR-0042", "location": "page:2", "source": "region_ocr", "category": "k"},
        {"type": "DATE_OF_BIRTH", "text": "28 February 1979", "location": "page:2", "source": "region_ocr", "category": "k"},
    ])

    # PAGE 3: Dark-theme Admin Console Figure (l) + Dashboard Figure (m) + Tiny text Figure (p) + Identity Record (n)
    img3 = Image.new("RGB", (PAGE_W, PAGE_H), color=(255, 255, 255))
    draw3 = ImageDraw.Draw(img3)

    draw3.text((120, 80), "SECTION 3: SYSTEM CONSOLES & TELEMETRY", fill=(10, 25, 47), font=font_h1)
    draw3.line([120, 140, PAGE_W - 120, 140], fill=(15, 23, 42), width=3)

    # Embed Figure (l): Dark Console Screenshot (scale_w=1400)
    admin_img, admin_tokens = create_admin_console_image(scale_w=1400)
    img3.paste(admin_img, (120, 160))
    figures_info["figure_l"] = {
        "page": 3,
        "type": "dark_console",
        "tokens": admin_tokens,
        "bbox": [120, 160, 120 + admin_img.width, 160 + admin_img.height],
    }

    # Add Figure (l) entities to ground truth
    admin_entities_data = [
        ("M. T. Achebe", "m.achebe@cadencecorp.net"),
        ("T. Lindqvist", "t.lindqvist@cadencecorp.net"),
        ("P. S. Raghavan", "p.raghavan@cadencecorp.net"),
        ("Kavita Reddy", "k.reddy@cadencecorp.net"),
        ("Arthur Pendelton", "a.pendelton@cadencecorp.net"),
        ("Beatrix Thorne", "b.thorne@cadencecorp.net"),
        ("Gideon Cross", "g.cross@cadencecorp.net"),
        ("Yuki Tanaka", "y.tanaka@cadencecorp.net"),
        ("Fiona MacLeod", "f.macleod@cadencecorp.net"),
        ("Damian O'Connor", "d.oconnor@cadencecorp.net"),
    ]
    for p_name, p_email in admin_entities_data:
        gt_entities.append({"type": "PERSON", "text": p_name, "location": "page:3", "source": "region_ocr", "category": "l"})
        gt_entities.append({"type": "EMAIL", "text": p_email, "location": "page:3", "source": "region_ocr", "category": "l"})

    # Embed Figure (m): Dashboard
    y_m = 160 + admin_img.height + 40
    dash_img, dash_tokens = create_dashboard_image()
    img3.paste(dash_img, (120, y_m))
    figures_info["figure_m"] = {
        "page": 3,
        "type": "dashboard",
        "tokens": dash_tokens,
        "bbox": [120, y_m, 120 + dash_img.width, y_m + dash_img.height],
    }
    gt_entities.extend([
        {"type": "PERSON", "text": "Cedric Vance", "location": "page:3", "source": "region_ocr", "category": "m"},
        {"type": "PERSON", "text": "Amina Qasim", "location": "page:3", "source": "region_ocr", "category": "m"},
        {"type": "PERSON", "text": "P. S. Raghavan", "location": "page:3", "source": "region_ocr", "category": "m"},
        {"type": "PERSON", "text": "Nicolette Dupuis", "location": "page:3", "source": "region_ocr", "category": "m"},
    ])

    # Embed Figure (n): Identity Record & Figure (p): Tiny text side-by-side
    y_n = y_m + dash_img.height + 40
    id_img, id_tokens = create_identity_record_image()
    img3.paste(id_img, (120, y_n))
    figures_info["figure_n"] = {
        "page": 3,
        "type": "identity_record",
        "tokens": id_tokens,
        "bbox": [120, y_n, 120 + id_img.width, y_n + id_img.height],
    }
    gt_entities.extend([
        {"type": "PERSON", "text": "Lukas Brauer", "location": "page:3", "source": "region_ocr", "category": "n"},
        {"type": "EMAIL", "text": "l.brauer@cadencecorp.net", "location": "page:3", "source": "region_ocr", "category": "n"},
        {"type": "PHONE", "text": "+49 69 5555 0177", "location": "page:3", "source": "region_ocr", "category": "n"},
        {"type": "EMPLOYEE_ID", "text": "EMP-38104", "location": "page:3", "source": "region_ocr", "category": "n"},
    ])

    tiny_img, tiny_tokens = create_tiny_text_image()
    img3.paste(tiny_img, (600, y_n + 50))
    figures_info["figure_p"] = {
        "page": 3,
        "type": "tiny_text_unverified",
        "tokens": tiny_tokens,
        "bbox": [600, y_n + 50, 600 + tiny_img.width, y_n + 50 + tiny_img.height],
        "expected_status": "UNVERIFIED",
    }
    # Deliberately note that Lysander Vance is tiny text (<7px), so it must NOT be claimed as read
    gt_entities.append({"type": "PERSON", "text": "Lysander Vance", "location": "page:3", "source": "region_ocr", "category": "p", "tricky": True})

    # Save pages to image-only PDF using PyMuPDF
    for p_img in [img1, img2, img3]:
        # Convert to JPEG/PNG bytes and insert into fitz Document
        buf = io.BytesIO()
        p_img.save(buf, format="PNG")
        buf.seek(0)
        img_bytes = buf.read()
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        page.insert_image(page.rect, stream=img_bytes)

    doc.save(str(pdf_path))
    doc.close()
    print(f"Generated {pdf_path} (3 pages, 1654x2339 px image-only)")

    return gt_entities, must_not_redact, figures_info


# ---------------------------------------------------------------------------
# Generator: 2. training_guide.docx
# ---------------------------------------------------------------------------
def generate_training_guide_docx() -> Tuple[List[Dict], List[str], Dict]:
    docx_path = OUT_DIR / "training_guide.docx"
    doc = Document()

    gt_entities = []
    must_not_redact = []
    figures_info = {}

    # Header / Title
    doc.add_heading("Cadence Compliance Training & Field Directory", 0)

    # Paragraph with label-word phrases and corporate header
    p_corp = doc.add_paragraph(
        "Cadence Financial Group | Corporate Data Privacy Office | Supervised by Board Risk Committee\n"
        "All employee identifiers and national identifiers must be safeguarded according to policy."
    )
    must_not_redact.extend([
        "employee identifiers",
        "national identifiers",
        "Board Risk Committee",
        "Data Privacy Office"
    ])

    # Section 1: Staff Table with Name directly before EMP, Email, Phone (no field label)
    doc.add_heading("1. Emergency Response Staff Contacts", level=1)
    p_intro = doc.add_paragraph(
        "The following designated leaders oversee global regional technical contingency operations. "
        "Each full name appears directly adjacent to employee credentials:"
    )

    table = doc.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Designated Leader"
    hdr_cells[1].text = "Official Email / Credentials"
    hdr_cells[2].text = "Direct Contact"

    staff_data = [
        ("M. T. Achebe", "m.achebe@cadencecorp.net (EMP-88210)", "+44 20 7555 0171"),
        ("T. Lindqvist", "t.lindqvist@cadencecorp.net (EMP-99341)", "+46 8 5555 0121"),
        ("P. S. Raghavan", "p.raghavan@cadencecorp.net (EMP-77124)", "+91 80 4555 0182"),
        ("Kavita Reddy", "k.reddy@cadencecorp.net (EMP-66219)", "+91 22 4555 0117"),
    ]

    for name, creds, phone in staff_data:
        row = table.add_row().cells
        row[0].text = name
        row[1].text = creds
        row[2].text = phone

        gt_entities.append({"type": "PERSON", "text": name, "location": "table:1", "source": "native", "category": "g"})
        email = creds.split()[0]
        emp_id = creds.split()[1].strip("()")
        gt_entities.append({"type": "EMAIL", "text": email, "location": "table:1", "source": "native", "category": "g"})
        gt_entities.append({"type": "EMPLOYEE_ID", "text": emp_id, "location": "table:1", "source": "native", "category": "g"})
        gt_entities.append({"type": "PHONE", "text": phone, "location": "table:1", "source": "native", "category": "g"})

    # Surname alone elsewhere in document (Issue 6 test)
    doc.add_heading("2. Delegation of Authority", level=1)
    doc.add_paragraph(
        "Authority is delegated to Achebe for UK incident sign-offs. "
        "Separately, Lindqvist retains operational oversight over European data transfers, "
        "while Raghavan provides final technical auditing approval."
    )
    gt_entities.extend([
        {"type": "PERSON", "text": "Achebe", "location": "body", "source": "native", "category": "g"},
        {"type": "PERSON", "text": "Lindqvist", "location": "body", "source": "native", "category": "g"},
        {"type": "PERSON", "text": "Raghavan", "location": "body", "source": "native", "category": "g"},
    ])

    # Embed Figure (l): Admin Console PNG at 600 px, 1500 px, and 2500 px widths
    doc.add_heading("3. IAM Console Snapshots", level=1)
    admin_img_600, _ = create_admin_console_image(scale_w=600)
    admin_img_1500, _ = create_admin_console_image(scale_w=1500)
    admin_img_2500, admin_tokens = create_admin_console_image(scale_w=2500)

    buf_600 = io.BytesIO()
    admin_img_600.save(buf_600, format="PNG")
    buf_600.seek(0)
    doc.add_paragraph("Console Overview (Compact 600px):")
    doc.add_picture(buf_600, width=Inches(3.0))

    buf_1500 = io.BytesIO()
    admin_img_1500.save(buf_1500, format="PNG")
    buf_1500.seek(0)
    doc.add_paragraph("Console Overview (Standard 1500px):")
    doc.add_picture(buf_1500, width=Inches(5.5))

    buf_2500 = io.BytesIO()
    admin_img_2500.save(buf_2500, format="PNG")
    buf_2500.seek(0)
    doc.add_paragraph("Console Overview (High-Res 2500px):")
    doc.add_picture(buf_2500, width=Inches(6.5))

    # Save initial docx
    doc.save(str(docx_path))

    # Now inject SVG into docx media archive: word/media/architecture_contact.svg
    svg_data = create_svg_content()
    # Add SVG entity to ground truth
    gt_entities.append({"type": "PERSON", "text": "Zachary Sterling", "location": "image:architecture_contact.svg", "source": "svg", "category": "o"})
    gt_entities.append({"type": "EMAIL", "text": "z.sterling@cadencecorp.net", "location": "image:architecture_contact.svg", "source": "svg", "category": "o"})

    # Re-zip docx with SVG file
    temp_zip = OUT_DIR / "temp_training_guide.docx"
    with zipfile.ZipFile(docx_path, "r") as zin:
        with zipfile.ZipFile(temp_zip, "w") as zout:
            for item in zin.infolist():
                buffer = zin.read(item.filename)
                zout.writestr(item, buffer)
            # Add SVG file into word/media/
            zout.writestr("word/media/architecture_contact.svg", svg_data.encode("utf-8"))

    if docx_path.exists():
        docx_path.unlink()
    temp_zip.rename(docx_path)

    print(f"Generated {docx_path} (with 3 PNGs and 1 SVG)")
    return gt_entities, must_not_redact, figures_info


# ---------------------------------------------------------------------------
# Generator: 3. org_pack.pptx
# ---------------------------------------------------------------------------
def generate_org_pack_pptx() -> Tuple[List[Dict], List[str], Dict]:
    pptx_path = OUT_DIR / "org_pack.pptx"
    prs = Presentation()
    prs.slide_width = PInches(13.333)
    prs.slide_height = PInches(7.5)

    gt_entities = []
    must_not_redact = []
    figures_info = {}

    blank_layout = prs.slide_layouts[6]

    # Slide 1: Cover & Organization
    slide1 = prs.slides.add_slide(blank_layout)
    txBox = slide1.shapes.add_textbox(PInches(1.0), PInches(0.8), PInches(11.0), PInches(1.5))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = "Cadence Global Operations & Organization Pack"
    p.font.size = PPt(36)
    p.font.bold = True

    p2 = tf.add_paragraph()
    p2.text = "Supervised by Board Risk Committee and Data Privacy Office"
    p2.font.size = PPt(20)

    # Add text box with address and identifiers
    txBox2 = slide1.shapes.add_textbox(PInches(1.0), PInches(2.5), PInches(11.0), PInches(3.5))
    tf2 = txBox2.text_frame
    p_info = tf2.paragraphs[0]
    p_info.text = "Regional Compliance Lead: Marcus Vance (SSN 900-55-2043)"
    p_info.font.size = PPt(18)

    p_addr = tf2.add_paragraph()
    p_addr.text = "Mailing address: home address 88 Willowbank Court, Hoboken, NJ 07030"
    p_addr.font.size = PPt(18)

    p_phone = tf2.add_paragraph()
    p_phone.text = "Direct Hotline: +44 20 7555 0171 | Singapore: +65 6555 0172"
    p_phone.font.size = PPt(18)

    gt_entities.extend([
        {"type": "PERSON", "text": "Marcus Vance", "location": "slide:1", "source": "native", "category": "b"},
        {"type": "SSN", "text": "900-55-2043", "location": "slide:1", "source": "native", "category": "b"},
        {"type": "ADDRESS", "text": "88 Willowbank Court, Hoboken, NJ 07030", "location": "slide:1", "source": "native", "category": "d"},
        {"type": "PHONE", "text": "+44 20 7555 0171", "location": "slide:1", "source": "native", "category": "a"},
        {"type": "PHONE", "text": "+65 6555 0172", "location": "slide:1", "source": "native", "category": "a"},
    ])

    # Slide 2: Embedded Admin Console Screenshot
    slide2 = prs.slides.add_slide(blank_layout)
    txBox3 = slide2.shapes.add_textbox(PInches(1.0), PInches(0.5), PInches(11.0), PInches(0.8))
    tf3 = txBox3.text_frame
    p3 = tf3.paragraphs[0]
    p3.text = "Live IAM Admin Console Telemetry"
    p3.font.size = PPt(28)
    p3.font.bold = True

    admin_img, admin_tokens = create_admin_console_image(scale_w=1200)
    buf_admin = io.BytesIO()
    admin_img.save(buf_admin, format="PNG")
    buf_admin.seek(0)
    slide2.shapes.add_picture(buf_admin, PInches(1.0), PInches(1.5), width=PInches(11.0))

    figures_info["figure_l_pptx"] = {
        "slide": 2,
        "type": "dark_console_pptx",
        "tokens": admin_tokens,
    }

    prs.save(str(pptx_path))
    print(f"Generated {pptx_path} (2 slides)")
    return gt_entities, must_not_redact, figures_info


# ---------------------------------------------------------------------------
# Main Generation Function
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("GENERATING SYNTHETIC REGRESSION PACK (/synthetic_regression)")
    print("=" * 70)

    pdf_ents, pdf_must_not, pdf_figs = generate_scanned_pdf()
    docx_ents, docx_must_not, docx_figs = generate_training_guide_docx()
    pptx_ents, pptx_must_not, pptx_figs = generate_org_pack_pptx()

    # Save Ground Truth JSONs
    # 1. scanned_policy.json
    gt_pdf = {
        "document": "scanned_policy.pdf",
        "entities": pdf_ents,
        "must_not_redact": pdf_must_not,
        "figures": pdf_figs,
    }
    with open(GT_DIR / "scanned_policy.json", "w", encoding="utf-8") as f:
        json.dump(gt_pdf, f, indent=2)

    # 2. training_guide.json
    gt_docx = {
        "document": "training_guide.docx",
        "entities": docx_ents,
        "must_not_redact": docx_must_not,
        "figures": docx_figs,
    }
    with open(GT_DIR / "training_guide.json", "w", encoding="utf-8") as f:
        json.dump(gt_docx, f, indent=2)

    # 3. org_pack.json
    gt_pptx = {
        "document": "org_pack.pptx",
        "entities": pptx_ents,
        "must_not_redact": pptx_must_not,
        "figures": pptx_figs,
    }
    with open(GT_DIR / "org_pack.json", "w", encoding="utf-8") as f:
        json.dump(gt_pptx, f, indent=2)

    # Also save unified regression ground truth summary for easy category inspection
    all_figures = {**pdf_figs, **docx_figs, **pptx_figs}
    all_must_not = sorted(list(set(pdf_must_not + docx_must_not + pptx_must_not)))
    manifest = {
        "description": "Synthetic Regression Pack Manifest",
        "must_not_redact": all_must_not,
        "figures": all_figures,
        "total_pdf_entities": len(pdf_ents),
        "total_docx_entities": len(docx_ents),
        "total_pptx_entities": len(pptx_ents),
    }
    with open(GT_DIR / "regression_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 70)
    print("REGRESSION PACK GENERATION COMPLETE.")
    print(f"Directory: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
