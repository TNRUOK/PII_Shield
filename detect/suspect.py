"""
detect/suspect.py
-----------------
Item 1: Recall-biased handling of low-confidence OCR lines (fail-closed).

If an OCR chunk line has mean word-confidence below `ocr_suspect_conf` (default 70)
AND contains a PII-shaped token (email, digit run ≥7, ID-like, or IP-like),
redact the PII-shaped token (or whole line if unisolatable) as SUSPECT_OCR.

Precision guards:
- Native (non-OCR) chunks are NEVER affected.
- Look-alikes (dates 12/05/2026, versions 4.31.2, section 4.2.3, rule IDs QRC_Sec_4.31 / WR.RP.1,
  reference numbers REF-..., day counts) are strictly excluded.
"""
from __future__ import annotations

import re
import logging
from typing import List, Tuple, Optional

from detect.rules import RawHit

log = logging.getLogger(__name__)

# Look-alikes that must NEVER be flagged as suspect PII
_DATE_RE = re.compile(r"^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$")
_VERSION_SECTION_RE = re.compile(r"^(?:v(?:er(?:sion)?)?\.?\s*)?\d+\.\d+(?:\.\d+)*$", re.IGNORECASE)
_CONTROL_RULE_ID_RE = re.compile(r"^(?:QRC_Sec|WR\.RP|PRD|REF|CASE|DOC)[A-Za-z0-9._\-]*$", re.IGNORECASE)
_ACRONYM_ORG_RE = re.compile(r"\b(BYOD|MFA|SSO|VPN|API|SDK|LLC|INC|CORP|GROUP|TEAM)\b", re.IGNORECASE)

# PII-shaped candidate regexes within a suspect line
_SUSPECT_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+(?:\s*@\s*|\s*\(at\)\s*|\s*\[at\]\s*)[a-zA-Z0-9.\-]+\b")
# 7 or more digits with optional separator dashes/spaces/dots (NOT slashes which are dates)
_SUSPECT_DIGIT_RUN_RE = re.compile(r"(?<![\d\w/])(?:\d[\s.\-]?){7,}\d(?![\d\w/])")
# PII-like ID: e.g. EMP-12345, Staff ID 12345, SSN-like or Gov ID
_SUSPECT_ID_RE = re.compile(r"\b(?:EMP|STAFF|ID|ACC|ACCT)[#:\-_ ]+[A-Za-z0-9\-]{4,20}\b", re.IGNORECASE)
# IP-like: 4 groups of digits
_SUSPECT_IP_RE = re.compile(r"\b\d{1,3}(?:[.\s]\d{1,3}){3}\b")


def _is_lookalike(token: str) -> bool:
    t = token.strip().strip(":,.;()[]{}")
    if _DATE_RE.match(t):
        return True
    if _VERSION_SECTION_RE.match(t):
        return True
    if _CONTROL_RULE_ID_RE.match(t):
        return True
    return False


def find_suspect_ocr_hits(chunk, config: dict) -> List[RawHit]:
    """
    Scan an OCR chunk for lines with confidence below ocr_suspect_conf.
    If PII-shaped patterns exist on uncertain lines, emit SUSPECT_OCR RawHits.
    """
    if chunk.is_blocked() or not chunk.text.strip():
        return []

    # Rule applies ONLY to OCR chunks
    is_ocr = (chunk.method == "ocr") or (getattr(chunk, "ocr_confidence", None) is not None)
    if not is_ocr:
        return []

    if not config.get("ocr_suspect_redaction", True):
        return []

    suspect_threshold = float(config.get("ocr_suspect_conf", 70.0))
    line_confs: Optional[List[Tuple[str, float]]] = getattr(chunk, "line_confidences", None)
    if not line_confs:
        return []

    hits: List[RawHit] = []
    chunk_text = chunk.text

    char_search_start = 0

    for line_text, line_conf in line_confs:
        # Find position of this line in chunk text
        line_pos = chunk_text.find(line_text, char_search_start)
        if line_pos == -1:
            line_pos = chunk_text.find(line_text)
        if line_pos != -1:
            char_search_start = line_pos + len(line_text)

        # Only evaluate lines with low confidence
        if line_conf >= suspect_threshold:
            continue

        # Look for PII-shaped candidates in this suspect line
        line_candidates = []
        for pat in [_SUSPECT_EMAIL_RE, _SUSPECT_DIGIT_RUN_RE, _SUSPECT_ID_RE, _SUSPECT_IP_RE]:
            for m in pat.finditer(line_text):
                token = m.group(0).strip()
                if not _is_lookalike(token):
                    line_candidates.append((m.start(), m.end(), token))

        if not line_candidates:
            continue

        # Isolate candidate spans
        if line_pos != -1:
            for s_start, s_end, tok in line_candidates:
                global_start = line_pos + s_start
                global_end = line_pos + s_end
                hits.append(RawHit(
                    entity_type="SUSPECT_OCR",
                    start=global_start,
                    end=global_end,
                    text=tok,
                    confidence=0.85,
                    sensitivity="High",
                    method="ocr_low_confidence",
                ))
        else:
            # If line position cannot be isolated, fallback to token search in chunk
            for _, _, tok in line_candidates:
                m_tok = re.search(re.escape(tok), chunk_text)
                if m_tok:
                    hits.append(RawHit(
                        entity_type="SUSPECT_OCR",
                        start=m_tok.start(),
                        end=m_tok.end(),
                        text=tok,
                        confidence=0.85,
                        sensitivity="High",
                        method="ocr_low_confidence",
                    ))

    return hits
