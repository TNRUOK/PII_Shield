"""
detect/rules.py
---------------
Layer A: Regex-based PII detection with per-type validators.

Each RULE entry is a dict:
  pattern     : compiled re.Pattern (IGNORECASE unless noted)
  entity_type : canonical type string
  validator   : optional callable(match_text) -> bool; if returns False the hit
                is either dropped (hard) or its confidence is reduced (soft).
  confidence  : base confidence for a match that passes validation
  soft_penalty: if validator returns False, subtract this from confidence
                (use 1.0 as hard reject, smaller for soft demote)
  sensitivity : "High" | "Medium" | "Low"
  value_group : optional int specifying capture group for the actual entity value
  ocr_only    : optional bool, only run on OCR chunks

run_rules(text, config, is_ocr=False) -> List[RawHit]
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable

from detect.validators import (
    luhn_check, verhoeff_check, iban_check, ssn_sanity, ipv4_check,
    phone_validate, pesel_check
)

log = logging.getLogger(__name__)


@dataclass
class RawHit:
    """A single regex detection hit before merging."""
    entity_type: str
    start: int
    end: int
    text: str                 # raw matched text (NOT included in report)
    confidence: float
    sensitivity: str
    method: str = "rule"


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

def _compile(pattern: str, flags: int = re.IGNORECASE) -> re.Pattern:
    return re.compile(pattern, flags)


# Look-alike / exclusion patterns for phone
_DATE_OR_VERSION_RE = re.compile(
    r"^(?:\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}|\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}|v?\d+\.\d+(?:\.\d+)*)$",
    re.IGNORECASE
)
_CONTROL_ID_RE = re.compile(r"^(?:QRC_Sec|WR\.RP|PRD|REF|CASE|DOC|POL|FORM)[A-Za-z0-9._\-]*$", re.IGNORECASE)

def _is_phone_exclusion(token: str) -> bool:
    clean = token.strip().strip(":,.;()[]{}")
    if _DATE_OR_VERSION_RE.match(clean):
        return True
    if _CONTROL_ID_RE.match(clean):
        return True
    # Count digits
    digits = re.sub(r"\D", "", clean)
    if len(digits) < 7 or len(digits) > 15:
        return True
    return False


# Street suffixes for Address rule
_STREET_SUFFIX = (
    r"(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Lane|Ln\.?|Drive|Dr\.?|Court|Ct\.?|"
    r"Crescent|Cres\.?|Boulevard|Blvd\.?|Way|Place|Pl\.?|Square|Sq\.?|Terrace|Ter\.?)"
)

# Known TLDs for email validation
_KNOWN_TLDS = r"(?:com|org|net|edu|gov|io|ai|co|uk|de|fr|in|jp|cn|ca|au|pl|se|ph|biz|info)"


RULES = [
    # --- Email ---
    {
        "pattern": _compile(
            r"<?[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,10}(?:\.[a-zA-Z]{2,6})?>?"
        ),
        "entity_type": "EMAIL",
        "validator": None,
        "confidence": 0.95,
        "soft_penalty": 0.0,
        "sensitivity": "Medium",
    },

    # --- OCR-tolerant email: literal @ with possible space OR explicit bracketed/spaced at ---
    # Only evaluated on OCR chunks (is_ocr=True). Forbids bare 'at' inside words.
    {
        "pattern": _compile(
            r"\b[a-zA-Z0-9._%+\-]+"
            r"(?:"
            r"\s*@\s*"
            r"|\s*\(at\)\s*"
            r"|\s*\[at\]\s*"
            r"|\s+at\s+"
            r")"
            r"[a-zA-Z0-9\-]+"
            r"(?:\s*\.\s*|\s*\[dot\]\s*|\s*\(dot\)\s*)"
            + _KNOWN_TLDS +
            r"\b"
        ),
        "entity_type": "EMAIL",
        "ocr_only": True,
        "validator": lambda t: ("@" in t or "(at)" in t.lower() or "[at]" in t.lower() or " at " in t.lower()),
        "confidence": 0.85,
        "soft_penalty": 0.0,
        "sensitivity": "Medium",
    },

    # --- International Phone: +<country code> followed by 7 to 14 digits ---
    # Trailing suffixes like ' (mobile)' are excluded by stopping at the last digit
    {
        "pattern": _compile(
            r"(?<![A-Za-z0-9])"
            r"\+[1-9]\d{0,3}"                                # +country code (1-4 digits)
            r"(?:[\s.\-]?\(?\d{1,5}\)?){2,6}"              # 7-14 further digits with separators
            r"(?!\d)"
        ),
        "entity_type": "PHONE",
        "validator": lambda t: phone_validate(t) and not _is_phone_exclusion(t),
        "confidence": 0.88,
        "soft_penalty": 0.0,
        "sensitivity": "Medium",
    },

    # --- US Phone: (NXX) NXX-XXXX, NXX-NXX-XXXX, NXX.NXX.XXXX, and optional extension ---
    {
        "pattern": _compile(
            r"(?<!\d)"
            r"(\+?1[\s.\-]?)?"                       # optional country code
            r"\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}"  # NXX sep NXX sep XXXX
            r"(?:\s*(?:x|ext\.?|extension)\s*\d{1,6})?"  # optional extension
            r"(?!\d)"
        ),
        "entity_type": "PHONE",
        "validator": None,
        "confidence": 0.80,
        "soft_penalty": 0.0,
        "sensitivity": "Medium",
    },

    # --- India Phone: +91 followed by 10 digits (e.g. +91 98765 43210) ---
    {
        "pattern": _compile(
            r"(?:^|(?<=[^\d+]))(?:\+91[\s.\-]?)?[6-9]\d{4}[\s.\-]?\d{5}(?!\d)"
        ),
        "entity_type": "PHONE",
        "validator": None,
        "confidence": 0.82,
        "soft_penalty": 0.0,
        "sensitivity": "Medium",
    },

    # --- US Phone: 10 contiguous digits (e.g. 4522327928) ---
    {
        "pattern": _compile(
            r"(?<![\d\w\-])([2-9]\d{2}[2-9]\d{6})(?![\d\w\-])"
        ),
        "entity_type": "PHONE",
        "validator": lambda t: not (t.startswith("20") and int(t[:4]) in range(1950, 2050)),
        "confidence": 0.78,
        "soft_penalty": 1.0,
        "sensitivity": "Medium",
    },

    # --- US SSN: NNN-NN-NNNN (standard) ---
    {
        "pattern": _compile(
            r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"
        ),
        "entity_type": "SSN",
        "validator": lambda t: ssn_sanity(t, allow_test_ranges=True),
        "confidence": 0.95,
        "soft_penalty": 1.0,   # hard reject if fails sanity
        "sensitivity": "High",
    },

    # --- Labelled SSN: preceded by SSN / social security (allowing 9xx area numbers) ---
    {
        "pattern": _compile(
            r"(?i:\b(?:ssn|social\s+security(?:\s+(?:no|number|#))?|ss#)\s*[:\-]?\s*)"
            r"(\d{3}[\s\-]\d{2}[\s\-]\d{4}|\d{9})\b"
        ),
        "entity_type": "SSN",
        "value_group": 1,
        "validator": lambda t: ssn_sanity(t, allow_test_ranges=True),
        "confidence": 0.95,
        "soft_penalty": 1.0,
        "sensitivity": "High",
    },

    # --- Narrative SSN Last Four Digits ---
    {
        "pattern": _compile(
            r"(?i:\b(?:last\s+four\s+digits|last\s+four|social\s+security|ssn)\b[^.\n]{0,60}?\b)\(?(\d{4})\)?"
        ),
        "entity_type": "SSN",
        "value_group": 1,
        "confidence": 0.90,
        "soft_penalty": 0.0,
        "sensitivity": "High",
    },

    # --- Credit / Debit Card: 13–19 digit groups (Luhn validated) ---
    {
        "pattern": _compile(
            r"(?<!\d)"
            r"(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))"  # major IINs
            r"[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}(?:\d{1,3})?"
            r"(?!\d)"
        ),
        "entity_type": "CREDIT_CARD",
        "validator": lambda t: luhn_check(re.sub(r"[\s\-]", "", t)),
        "confidence": 0.93,
        "soft_penalty": 1.0,
        "sensitivity": "High",
    },

    # --- Account Number (labelled) ---
    {
        "pattern": _compile(
            r"(?:account\s+(?:no|number|#)|a/?c\s+no)\s*[:\-]?\s*(\d[\d\s\-]{6,20}\d)"
        ),
        "entity_type": "ACCOUNT_NUMBER",
        "validator": None,
        "confidence": 0.85,
        "soft_penalty": 0.0,
        "sensitivity": "High",
    },

    # --- Card / Account Ending Fragment ---
    {
        "pattern": _compile(
            r"(?i:\b(?:card|savings\s+account|account)\s+ending\s+\d{4}\b)"
        ),
        "entity_type": "ACCOUNT_NUMBER",
        "confidence": 0.88,
        "soft_penalty": 0.0,
        "sensitivity": "High",
    },

    # --- IBAN ---
    {
        "pattern": _compile(
            r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b",
            re.IGNORECASE,
        ),
        "entity_type": "IBAN",
        "validator": iban_check,
        "confidence": 0.90,
        "soft_penalty": 1.0,
        "sensitivity": "High",
    },

    # --- Passport (labelled context required OR standard format) ---
    {
        "pattern": _compile(
            r"(?:passport(?:\s+(?:no|number|#))?\s*[:\-]?\s*)?[A-Z]{1,2}\d{6,9}\b"
        ),
        "entity_type": "PASSPORT",
        "validator": None,
        "confidence": 0.75,
        "soft_penalty": 0.0,
        "sensitivity": "High",
    },

    # --- Date of Birth (labelled) ---
    {
        "pattern": _compile(
            r"(?i:\b(?:d\.?o\.?b\.?|date\s+of\s+birth|birth\s+date|born)\s*[:\-,]?\s*)"
            r"("
            r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}"
            r"|\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}"
            r"|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}"
            r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}"
            r")\b"
        ),
        "entity_type": "DATE_OF_BIRTH",
        "value_group": 1,
        "confidence": 0.90,
        "soft_penalty": 0.0,
        "sensitivity": "High",
    },

    # --- Date of Birth on line after 'DATE OF BIRTH' label in badge/form ---
    {
        "pattern": _compile(
            r"(?i:\bDATE\s+OF\s+BIRTH\b\s*[\n\r]+\s*)"
            r"("
            r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}"
            r"|\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}"
            r")\b"
        ),
        "entity_type": "DATE_OF_BIRTH",
        "value_group": 1,
        "confidence": 0.90,
        "soft_penalty": 0.0,
        "sensitivity": "High",
    },

    # --- Home Address (narrative introduction: home address, residential address, lives at) ---
    {
        "pattern": _compile(
            r"(?i:\b(?:home\s+address|residential\s+address|lives\s+at|to\s+your\s+home\s+at)\s*[:\-,]?\s*)"
            r"([0-9]{1,5}\s+[A-Za-z0-9\s,.\-#]+?\b" + _STREET_SUFFIX + r"[A-Za-z0-9\s,.\-#]*?\b\d{5}\b)"
        ),
        "entity_type": "ADDRESS",
        "value_group": 1,
        "confidence": 0.90,
        "soft_penalty": 0.0,
        "sensitivity": "High",
    },

    # --- Standard Street Address (number + street suffix + state/ZIP) ---
    {
        "pattern": _compile(
            r"\b([0-9]{1,5}\s+(?:[A-Z][a-z]+\s+){1,4}" + _STREET_SUFFIX +
            r"(?:,?\s+(?:Apartment|Apt\.?|Suite|Unit|Flat)\s+[0-9A-Za-z\-]+)?"
            r"(?:,?\s+[A-Z][a-z]+)?"
            r"(?:,?\s+(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\s+\d{5}(?:-\d{4})?))\b"
        ),
        "entity_type": "ADDRESS",
        "value_group": 1,
        "confidence": 0.85,
        "soft_penalty": 0.0,
        "sensitivity": "High",
    },

    # --- Aadhaar: 4-4-4 digit groups (requires Verhoeff OR Aadhaar label within 25 chars) ---
    {
        "pattern": _compile(
            r"(?<![0-9])(\d{4}[\s\-]\d{4}[\s\-]\d{4})(?![0-9])"
        ),
        "entity_type": "AADHAAR",
        "value_group": 1,
        "validator": lambda t: verhoeff_check(re.sub(r"[\s\-]", "", t)),
        "confidence": 0.88,
        "soft_penalty": 0.20,  # soft: penalty if checksum fails
        "sensitivity": "High",
    },

    # --- Aadhaar: bare 12 contiguous digits ---
    {
        "pattern": _compile(
            r"(?<![0-9])\d{12}(?![0-9])"
        ),
        "entity_type": "AADHAAR",
        "validator": verhoeff_check,
        "confidence": 0.70,
        "soft_penalty": 1.0,   # hard-reject bare 12 digits if Verhoeff fails
        "sensitivity": "High",
    },

    # --- PAN (India): AAAAA9999A pattern ---
    {
        "pattern": _compile(
            r"(?<![A-Z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Z0-9])"
        ),
        "entity_type": "PAN",
        "validator": None,
        "confidence": 0.88,
        "soft_penalty": 0.0,
        "sensitivity": "Medium",
    },

    # --- Director ID: DIR-NNNN ---
    {
        "pattern": _compile(
            r"\bDIR-\d{4}\b"
        ),
        "entity_type": "DIRECTOR_ID",
        "validator": None,
        "confidence": 0.90,
        "soft_penalty": 0.0,
        "sensitivity": "Medium",
    },

    # --- Vendor ID: MER-XX-NNNN ---
    {
        "pattern": _compile(
            r"\bMER-[A-Z]{2}-?\d{3,4}\b"
        ),
        "entity_type": "VENDOR_ID",
        "validator": None,
        "confidence": 0.90,
        "soft_penalty": 0.0,
        "sensitivity": "Medium",
    },

    # --- Polish PESEL ---
    {
        "pattern": _compile(
            r"(?i:\bPESEL\s*[:\-]?\s*)(\d{11}|\d{6}[xX*]{5})\b"
        ),
        "entity_type": "PESEL",
        "value_group": 1,
        "validator": pesel_check,
        "confidence": 0.92,
        "soft_penalty": 0.0,
        "sensitivity": "High",
    },

    # --- Employee ID: labelled or EMP-NNNN style ---
    {
        "pattern": _compile(
            r"(?:"
            r"(?:\bemployee\s+id|\bemp(?:loyee)?\s*(?:no|number|#)|\bstaff\s+id)"
            r"\s*[:\-]\s*([A-Z0-9\-]{3,15})"  # labelled form requires separator : or -
            r"|"
            r"\bEMP[\-_]\d{3,8}\b"              # EMP-12345 style
            r")"
        ),
        "entity_type": "EMPLOYEE_ID",
        "validator": lambda t: any(c.isdigit() for c in t),
        "confidence": 0.88,
        "soft_penalty": 0.0,
        "sensitivity": "Medium",
    },

    # --- Generic Government ID (labelled: TIN, NID, National ID - word boundary required) ---
    {
        "pattern": _compile(
            r"(?i:\b(?:national\s+id|(?:tax(?:payer)?\s+)?(?:identification|id)\s+(?:no|number)|\btin\b|\bnid\b)"
            r"\s*[:\-]\s*([A-Z0-9][\w\-]{4,20}))"
        ),
        "entity_type": "GOVERNMENT_ID",
        "value_group": 1,
        "validator": lambda t: any(c.isdigit() for c in t) and len(t.strip()) >= 5 and t.lower() not in ("contact", "policy", "review"),
        "confidence": 0.80,
        "soft_penalty": 0.0,
        "sensitivity": "Low",
    },

    # --- IPv4 address ---
    {
        "pattern": _compile(
            r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"
        ),
        "entity_type": "IP_ADDRESS",
        "validator": ipv4_check,
        "confidence": 0.90,
        "soft_penalty": 1.0,
        "sensitivity": "Low",
    },

    # --- IPv4 labelled: 'IP:' or 'IP address:' context boost ---
    {
        "pattern": _compile(
            r"(?i:ip(?:\s+address)?(?:\s*\([^)]*\))?|host|workstation\s+ip)\s*[:\.\-]\s*"
            r"([0-9]{1,3}(?:[.\s][0-9]{1,3}){3})\b"
        ),
        "entity_type": "IP_ADDRESS",
        "value_group": 1,
        "validator": ipv4_check,
        "confidence": 0.92,
        "soft_penalty": 1.0,
        "sensitivity": "Low",
    },

    # --- IPv6 address: full 8-group form ---
    {
        "pattern": _compile(
            r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
        ),
        "entity_type": "IP_ADDRESS",
        "validator": None,
        "confidence": 0.88,
        "soft_penalty": 0.0,
        "sensitivity": "Low",
    },

    # --- IPv6 address: compressed form with :: (e.g. ::1, fe80::1, 2001:db8::1) ---
    {
        "pattern": _compile(
            r"\b(?:[0-9a-fA-F]{0,4}:){2,7}(?::[0-9a-fA-F]{0,4}|[0-9a-fA-F]{1,4})\b"
            r"|(?<![a-zA-Z0-9])::1\b"
        ),
        "entity_type": "IP_ADDRESS",
        "validator": None,
        "confidence": 0.82,
        "soft_penalty": 0.0,
        "sensitivity": "Low",
    },

    # --- OCR-tolerant email: (at)/[at]/(AT)/[dot] forms and spaced @ ---
    # Only evaluated on OCR chunks (is_ocr=True) so native precision does not drop.
    # Requires an explicit email signal (@, (at), [at]) AND a known TLD-like suffix.
    {
        "pattern": _compile(
            r"[a-zA-Z0-9._%+\-]+"
            r"(?:\s*@\s*|\s*\(at\)\s*|\s*\[at\]\s*|\s*AT\s+)"
            r"[a-zA-Z0-9.\-]+"
            r"(?:\s*\.\s*|\s*\[dot\]\s*|\s*\(dot\)\s*)"
            + _KNOWN_TLDS +
            r"\b"
        ),
        "entity_type": "EMAIL",
        "ocr_only": True,
        "validator": lambda t: (
            "@" in t
            or "(at)" in t.lower()
            or "[at]" in t.lower()
            or " at " in t.lower()
        ),
        "confidence": 0.82,
        "soft_penalty": 0.0,
        "sensitivity": "Medium",
    },
]


# ---------------------------------------------------------------------------
# Corporate address check helper
# ---------------------------------------------------------------------------

def _is_corporate_address_line(text: str, start: int, end: int, config: dict) -> bool:
    """Return True if the matched address line appears to be a corporate HQ address."""
    # Find start and end of the line
    line_start = text.rfind("\n", 0, start)
    line_start = 0 if line_start == -1 else line_start + 1
    line_end = text.find("\n", end)
    line_end = len(text) if line_end == -1 else line_end
    line_text = text[line_start:line_end].lower()

    # If personal label present, do NOT skip
    if re.search(r"\b(home|residential|lives\s+at|personal)\b", line_text):
        return False

    allowlist = config.get("corporate_address_allowlist", [
        "Inc.", "Ltd", "LLC", "Financial Group", "Corp", "Corporation", "AcmeCorp"
    ])
    for term in allowlist:
        if term.lower() in line_text:
            return True
    return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_rules(text: str, config: dict, is_ocr: bool = False) -> List[RawHit]:
    """
    Apply all enabled rules to `text` and return a list of RawHit objects.

    Parameters
    ----------
    text   : The chunk text to scan.
    config : Loaded config dict (used for enabled_entities and thresholds).
    is_ocr : Whether this text was extracted via OCR (enables OCR-tolerant rules).
    """
    enabled: set = set(config.get("enabled_entities", [e["entity_type"] for e in RULES]))
    profile_name = config.get("active_profile", "balanced")
    min_conf: float = config["profiles"][profile_name]["rule_confidence_threshold"]

    hits: List[RawHit] = []

    for rule in RULES:
        if rule.get("ocr_only") and not is_ocr:
            continue

        etype = rule["entity_type"]
        if etype not in enabled:
            continue

        for m in rule["pattern"].finditer(text):
            if "value_group" in rule and rule["value_group"] and rule["value_group"] <= len(m.groups()):
                raw = m.group(rule["value_group"])
                start = m.start(rule["value_group"])
                end = m.end(rule["value_group"])
            else:
                raw = m.group(0)
                start = m.start()
                end = m.end()

            if not raw or not raw.strip():
                continue

            # Corporate address filter
            if etype == "ADDRESS" and _is_corporate_address_line(text, start, end, config):
                log.debug("Skipping corporate address at [%d:%d]: %r", start, end, raw)
                continue

            # Aadhaar date/phone overlap veto
            if etype == "AADHAAR":
                context_prefix = text[max(0, start - 30):start]
                if re.search(r"(?:drafted|dated|review|version|\b20\d\d\b)\s*$", context_prefix, re.IGNORECASE):
                    # Likely a date followed by numbers, not Aadhaar
                    continue
                # Also check if preceded by Aadhaar/UID label
                has_label = bool(re.search(r"(?i:\b(aadhaar|uid)\b)", context_prefix))
                # If no label and Verhoeff check failed, skip
                if not has_label and not verhoeff_check(re.sub(r"[\s\-]", "", raw)):
                    continue

            conf = rule["confidence"]
            penalty = rule.get("soft_penalty", 0.0)

            # Run validator if provided
            if rule.get("validator") is not None:
                try:
                    valid = rule["validator"](raw)
                except Exception as exc:
                    log.debug("Validator error for %s (span [%d:%d]): %s", etype, start, end, exc)
                    valid = False

                if not valid:
                    if penalty >= 1.0:
                        # Hard reject
                        log.debug("Hard-rejected %s match (span [%d:%d]): %r", etype, start, end, raw)
                        continue
                    else:
                        conf -= penalty
                        log.debug("Soft-penalised %s match (span [%d:%d]) → conf %.2f", etype, start, end, conf)

            if conf < min_conf:
                continue

            hits.append(RawHit(
                entity_type=etype,
                start=start,
                end=end,
                text=raw,
                confidence=min(conf, 1.0),
                sensitivity=rule["sensitivity"],
            ))

    # Overlap deduplication: for hits of the same entity type, keep the widest span
    hits_sorted = sorted(hits, key=lambda h: (h.end - h.start), reverse=True)
    deduped: List[RawHit] = []
    for h in hits_sorted:
        if not any(h.entity_type == d.entity_type and d.start <= h.start and d.end >= h.end for d in deduped):
            deduped.append(h)

    # Sort in document offset order
    deduped.sort(key=lambda h: h.start)
    return deduped
