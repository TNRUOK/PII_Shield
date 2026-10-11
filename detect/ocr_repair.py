"""
detect/ocr_repair.py
--------------------
Item 2: Character-confusion candidate normalisation for OCR text.

OCR engines commonly confuse similar-looking characters, e.g.:
  O/o → 0,  I/l/| → 1,  S/$ → 5,  B → 8,  Z → 2,  G → 6

This module:
1. Applies those substitutions to candidate tokens from OCR chunks ONLY.
2. Validates the repaired candidates using checksum validators.
3. Emits RawHit objects over the ORIGINAL span in the original text.
4. Also repairs common TLD typos in email addresses (.cam→.com, .corn→.com, rn→m).

Precision guards:
- Only runs on OCR chunks (method == "ocr" or ocr_confidence set).
- Skips dates, version numbers, section numbers, rule/control IDs.
- Cap at 5 repaired candidates per token.
- Never flags look-alike non-PII tokens.
"""
from __future__ import annotations

import re
import logging
from typing import List, Tuple, Optional

from detect.rules import RawHit
from detect.validators import luhn_check, verhoeff_check, ssn_sanity, ipv4_check

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Look-alike guard patterns (same as in suspect.py — avoid circular import)
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"^(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[/\-]\d{1,2}[/\-]\d{1,2})$")
_VERSION_SECTION_RE = re.compile(r"^(?:v(?:er(?:sion)?)?\.?\s*)?\d+\.\d+(?:\.\d+)*$", re.IGNORECASE)
_SECTION_NUM_RE = re.compile(r"^\d+(?:\.\d+)+$")
_CONTROL_RULE_ID_RE = re.compile(r"^(?:QRC_Sec|WR\.RP|PRD|REF|CASE|DOC)[A-Za-z0-9._\-]*$", re.IGNORECASE)
_SHORT_INT_RE = re.compile(r"^\d{1,6}$")


def _is_lookalike(token: str) -> bool:
    t = token.strip().strip(":,.;()[]{}").rstrip("?/")
    if _DATE_RE.match(t):
        return True
    if _VERSION_SECTION_RE.match(t):
        return True
    if _SECTION_NUM_RE.match(t):
        return True
    if _CONTROL_RULE_ID_RE.match(t):
        return True
    if _SHORT_INT_RE.match(t):
        return True
    return False


# ---------------------------------------------------------------------------
# Character confusion substitution maps
# ---------------------------------------------------------------------------

# Applied to digit-run candidates only (token that looks numeric after repair)
_OCR_DIGIT_SUBS: List[Tuple[str, str]] = [
    # Order matters: multi-char before single-char
    ("rn", "m"),   # for email TLD repair leakover — handled separately
    ("O", "0"),
    ("o", "0"),
    ("I", "1"),
    ("l", "1"),
    ("|", "1"),
    ("S", "5"),
    ("$", "5"),
    ("B", "8"),
    ("Z", "2"),
    ("G", "6"),
]

# Characters to strip from inside digit runs (noise artefacts)
_DIGIT_NOISE_RE = re.compile(r"[/? ]")

# Token patterns for potential numeric PII
_NUMERIC_CANDIDATE_RE = re.compile(
    r"\b[0-9OoIlSBZG$|]{6,}\b"  # 6+ chars that could be a digit run after repair
)
_SSN_CANDIDATE_RE = re.compile(
    r"\b[0-9OoIlSBZG$|]{3}-[0-9OoIlSBZG$|]{2}-[0-9OoIlSBZG$|]{4}\b"
)

# Email pattern (to detect and fix TLD typos)
_EMAIL_CANDIDATE_RE = re.compile(
    r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\b"
)

# TLD repair map: OCR-garbled TLD → correct TLD
_TLD_REPAIR = {
    ".cam": ".com",
    ".corn": ".com",
    ".coni": ".com",
    ".con": ".com",
    ".org0": ".org",
    ".nef": ".net",
    ".nel": ".net",
}

# Phone-like: 10+ digit run (after repair)
_PHONE_MIN_DIGITS = 10
_CC_MIN_DIGITS = 13  # credit card
_SSN_DIGITS = 9

# Max repaired candidates to emit per token (precision guard)
_MAX_CANDIDATES_PER_TOKEN = 5


# ---------------------------------------------------------------------------
# Helper: apply substitutions to produce a "repaired" digit-only string
# ---------------------------------------------------------------------------

def _repair_digit_token(token: str) -> str:
    """Apply OCR character-confusion substitutions to produce a digit string."""
    result = token
    for src, dst in _OCR_DIGIT_SUBS:
        if src in ("rn",):   # only for email — skip here
            continue
        result = result.replace(src, dst)
    # Strip noise characters inside digit runs
    result = _DIGIT_NOISE_RE.sub("", result)
    return result


def _repair_email_tld(email: str) -> str:
    """Repair common OCR TLD confusions in an email address."""
    # rn → m at end of domain (e.g. .corn → .com already in _TLD_REPAIR, rn at word end)
    repaired = re.sub(r"rn\b", "m", email)
    for bad, good in _TLD_REPAIR.items():
        repaired = repaired.replace(bad, good)
    return repaired


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def find_ocr_repair_hits(chunk, config: dict) -> List[RawHit]:
    """
    Scan an OCR chunk for character-confusion mangled PII.
    Returns RawHit objects over the ORIGINAL span in chunk.text.
    Only runs on OCR chunks; returns [] for native text.
    """
    if chunk.is_blocked() or not chunk.text.strip():
        return []

    is_ocr = (chunk.method == "ocr") or (getattr(chunk, "ocr_confidence", None) is not None)
    if not is_ocr:
        return []

    text = chunk.text
    hits: List[RawHit] = []

    # --- Numeric candidate repair ---
    candidate_count = 0
    for m in _NUMERIC_CANDIDATE_RE.finditer(text):
        if candidate_count >= _MAX_CANDIDATES_PER_TOKEN:
            break
        raw_token = m.group(0)
        if _is_lookalike(raw_token):
            continue
        repaired = _repair_digit_token(raw_token)
        if not repaired.isdigit():
            continue
        if repaired == raw_token:
            continue  # No change — already handled by the normal rules layer
        n_digits = len(repaired)

        entity_type = None
        confidence = 0.0

        if n_digits == _SSN_DIGITS and (ssn_sanity(repaired) or ssn_sanity(f"{repaired[:3]}-{repaired[3:5]}-{repaired[5:]}")):
            entity_type = "SSN"
            confidence = 0.72   # lower because it's repaired OCR
        elif n_digits in (13, 14, 15, 16, 17, 18, 19) and luhn_check(repaired):
            entity_type = "CREDIT_CARD"
            confidence = 0.75
        elif n_digits == 12 and verhoeff_check(repaired):
            entity_type = "AADHAAR"
            confidence = 0.75
        elif n_digits >= _PHONE_MIN_DIGITS:
            entity_type = "PHONE"
            confidence = 0.65

        if entity_type:
            log.debug(
                "OCR repair: detected as %s (span [%d:%d])",
                entity_type, m.start(), m.end()
            )
            hits.append(RawHit(
                entity_type=entity_type,
                start=m.start(),
                end=m.end(),
                text=raw_token,
                confidence=confidence,
                sensitivity="High",
                method="ocr_repair",
            ))
            candidate_count += 1

    # --- SSN hyphenated candidate repair ---
    for m in _SSN_CANDIDATE_RE.finditer(text):
        raw_token = m.group(0)
        if _is_lookalike(raw_token):
            continue
        repaired = _repair_digit_token(raw_token)
        if repaired == raw_token:
            continue
        if ssn_sanity(repaired):
            log.debug(
                "OCR repair: detected as SSN (span [%d:%d])",
                m.start(), m.end()
            )
            hits.append(RawHit(
                entity_type="SSN",
                start=m.start(),
                end=m.end(),
                text=raw_token,
                confidence=0.72,
                sensitivity="High",
                method="ocr_repair",
            ))

    # --- Email TLD repair ---
    for m in _EMAIL_CANDIDATE_RE.finditer(text):
        raw_email = m.group(0)
        repaired_email = _repair_email_tld(raw_email)
        if repaired_email == raw_email:
            continue  # No change
        # Validate: must have a proper domain with a dot
        if "@" not in repaired_email or "." not in repaired_email.split("@")[-1]:
            continue
        log.debug(
            "OCR email TLD repair (span [%d:%d])",
            m.start(), m.end()
        )

        hits.append(RawHit(
            entity_type="EMAIL",
            start=m.start(),
            end=m.end(),
            text=raw_email,
            confidence=0.80,
            sensitivity="Medium",
            method="ocr_repair_email",
        ))

    return hits
