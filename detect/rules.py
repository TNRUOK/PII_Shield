"""
detect/rules.py
---------------
Layer A: Regex-based PII detection with per-type validators.

Each RULE entry is a dict:
  pattern   : compiled re.Pattern (IGNORECASE unless noted)
  entity_type: canonical type string
  validator  : optional callable(match_text) -> bool; if returns False the hit
               is either dropped (hard) or its confidence is reduced (soft).
  confidence : base confidence for a match that passes validation
  soft_penalty: if validator returns False, subtract this from confidence
                (use 1.0 as hard reject, smaller for soft demote)
  sensitivity: "High" | "Medium" | "Low"

run_rules(text, config) -> List[RawHit]
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable

from detect.validators import luhn_check, verhoeff_check, iban_check, ssn_sanity, ipv4_check

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


RULES = [
    # --- Email ---
    {
        "pattern": _compile(
            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
        ),
        "entity_type": "EMAIL",
        "validator": None,
        "confidence": 0.95,
        "soft_penalty": 0.0,
        "sensitivity": "Medium",
    },

    # --- US Phone: (NXX) NXX-XXXX or NXX-NXX-XXXX or 10 contiguous digits ---
    {
        "pattern": _compile(
            r"(?<!\d)"
            r"(\+?1[\s.\-]?)?"                     # optional country code
            r"\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]\d{4}"
            r"(?!\d)"
        ),
        "entity_type": "PHONE",
        "validator": None,
        "confidence": 0.80,
        "soft_penalty": 0.0,
        "sensitivity": "Medium",
    },

    # --- India Phone: +91 followed by 10 digits ---
    {
        "pattern": _compile(
            r"(?<!\d)(\+91[\s\-]?|0)?[6-9]\d{9}(?!\d)"
        ),
        "entity_type": "PHONE",
        "validator": None,
        "confidence": 0.82,
        "soft_penalty": 0.0,
        "sensitivity": "Medium",
    },

    # --- US SSN: NNN-NN-NNNN ---
    {
        "pattern": _compile(
            r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"
        ),
        "entity_type": "SSN",
        "validator": ssn_sanity,
        "confidence": 0.95,
        "soft_penalty": 1.0,   # hard reject if fails sanity
        "sensitivity": "High",
    },

    # --- Credit / Debit Card: 13–19 digit groups (Luhn validated) ---
    # Pattern is intentionally broad; Luhn filters false positives
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
        "soft_penalty": 1.0,   # hard reject if Luhn fails
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

    # --- Aadhaar: 4-4-4 digit groups (Verhoeff soft check) ---
    {
        "pattern": _compile(
            r"(?<![0-9])\d{4}[\s\-]\d{4}[\s\-]\d{4}(?![0-9])"
        ),
        "entity_type": "AADHAAR",
        "validator": lambda t: verhoeff_check(re.sub(r"[\s\-]", "", t)),
        "confidence": 0.85,
        "soft_penalty": 0.20,  # soft: reduce by 0.20 if checksum fails
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

    # --- Employee ID: labelled or EMP-NNNN style ---
    {
        "pattern": _compile(
            r"(?:"
            r"(?:employee\s+id|emp(?:loyee)?\s*(?:no|number|#)|staff\s+id)"
            r"\s*[:\-]?\s*([A-Z0-9\-]{3,15})"  # labelled form
            r"|"
            r"\bEMP[\-_]\d{3,8}\b"              # EMP-12345 style
            r")"
        ),
        "entity_type": "EMPLOYEE_ID",
        "validator": None,
        "confidence": 0.88,
        "soft_penalty": 0.0,
        "sensitivity": "Medium",
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

    # --- Date of Birth (labelled) ---
    {
        "pattern": _compile(
            r"(?:d(?:ate\s+of\s+)?o(?:f\s+)?b(?:irth)?|dob|birth\s+date)"
            r"\s*[:\-]?\s*"
            r"(\d{1,2}[\-/\.]\d{1,2}[\-/\.]\d{2,4}"
            r"|\d{4}[\-/\.]\d{1,2}[\-/\.]\d{1,2}"
            r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{1,2},?\s+\d{4})"
        ),
        "entity_type": "DATE_OF_BIRTH",
        "validator": None,
        "confidence": 0.85,
        "soft_penalty": 0.0,
        "sensitivity": "Medium",
    },

    # --- Generic Government ID (labelled: SSN, TIN, NID, National ID) ---
    {
        "pattern": _compile(
            r"(?:(?:national\s+id|(?:tax(?:payer)?\s+)?(?:identification|id)\s+(?:no|number)|tin|nid)"
            r"\s*[:\-]?\s*([A-Z0-9][\w\-]{4,20}))"
        ),
        "entity_type": "GOVERNMENT_ID",
        "validator": None,
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

    # --- IPv6 address (simplified) ---
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
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_rules(text: str, config: dict) -> List[RawHit]:
    """
    Apply all enabled rules to `text` and return a list of RawHit objects.

    Parameters
    ----------
    text   : The chunk text to scan.
    config : Loaded config dict (used for enabled_entities and thresholds).
    """
    enabled: set = set(config.get("enabled_entities", [e["entity_type"] for e in RULES]))
    profile_name = config.get("active_profile", "balanced")
    min_conf: float = config["profiles"][profile_name]["rule_confidence_threshold"]

    hits: List[RawHit] = []

    for rule in RULES:
        etype = rule["entity_type"]
        if etype not in enabled:
            continue

        for m in rule["pattern"].finditer(text):
            raw = m.group(0)
            conf = rule["confidence"]
            penalty = rule.get("soft_penalty", 0.0)

            # Run validator if provided
            if rule["validator"] is not None:
                try:
                    valid = rule["validator"](raw)
                except Exception as exc:
                    log.debug("Validator error for %s on %r: %s", etype, raw, exc)
                    valid = False

                if not valid:
                    if penalty >= 1.0:
                        # Hard reject
                        log.debug("Hard-rejected %s match: %r", etype, raw[:30])
                        continue
                    else:
                        conf -= penalty
                        log.debug("Soft-penalised %s match: %r → conf %.2f", etype, raw[:30], conf)

            if conf < min_conf:
                continue

            hits.append(RawHit(
                entity_type=etype,
                start=m.start(),
                end=m.end(),
                text=raw,
                confidence=min(conf, 1.0),
                sensitivity=rule["sensitivity"],
            ))

    return hits
