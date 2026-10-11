"""
detect/suspect.py
-----------------
Item 1 (v5): Fail-closed redaction of low-confidence OCR lines — gap-only.

If an OCR chunk line has mean word-confidence below `ocr_suspect_conf` (default 70)
AND contains a PII-shaped token (email, digit run ≥7, ID-like, or IP-like)
AND the token span is NOT already covered by a typed hit (PHONE, EMAIL, SSN, …),
redact the token as SUSPECT_OCR.

Precision guards:
- Native (non-OCR) chunks are NEVER affected.
- Look-alikes (dates 12/05/2026, versions 4.31.2, section 4.2.3, rule IDs QRC_Sec_4.31 /
  WR.RP.1, reference numbers REF-..., day counts ≤3 digits, bare integers ≤6 digits
  without surrounding context) are strictly excluded.
- Any span already covered by an existing typed hit is silently skipped.
"""
from __future__ import annotations

import re
import logging
from typing import List, Tuple, Optional

from detect.rules import RawHit

log = logging.getLogger(__name__)

_DATE_RE = re.compile(r"^(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[/\-]\d{1,2}[/\-]\d{1,2})$")
_VERSION_SECTION_RE = re.compile(r"^(?:v(?:er(?:sion)?)?\.?\s*)?\d+\.\d+(?:\.\d+)*$", re.IGNORECASE)
# Section numbers like 4.2.3 or 2.4 (digits separated by dots only)
_SECTION_NUM_RE = re.compile(r"^\d+(?:\.\d+)+$")
_CONTROL_RULE_ID_RE = re.compile(r"^(?:QRC_Sec|WR\.RP|PRD|REF|CASE|DOC|CL)[A-Za-z0-9._\-]*$", re.IGNORECASE)
_ACRONYM_ORG_RE = re.compile(r"\b(BYOD|MFA|SSO|VPN|API|SDK|LLC|INC|CORP|GROUP|TEAM)\b", re.IGNORECASE)
# Bare short integers ≤6 digits (day counts, page numbers, ref codes ≤6 digits)
_SHORT_INT_RE = re.compile(r"^\d{1,6}$")

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
    if _SECTION_NUM_RE.match(t):
        return True
    if _CONTROL_RULE_ID_RE.match(t):
        return True
    if _SHORT_INT_RE.match(t):
        return True  # bare ≤6-digit integers are too ambiguous (day counts, page nums)
    return False


def find_suspect_ocr_hits(
    chunk,
    config: dict,
    existing_hits: Optional[List] = None,
) -> List[RawHit]:
    """
    Scan an OCR chunk for lines with confidence below ocr_suspect_conf.
    If PII-shaped patterns exist on uncertain lines emit SUSPECT_OCR RawHits,
    BUT only for spans NOT already covered by an existing typed hit.

    Parameters
    ----------
    chunk         : The OCR Chunk being analysed.
    config        : Loaded config dict.
    existing_hits : RawHit list already collected from rules/NER/context layers.
                    Any candidate that overlaps an existing hit is skipped so that
                    typed entities (PHONE, EMAIL, …) always take priority.
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
                if _is_lookalike(token):
                    continue
                # Also check enclosing whitespace-delimited word in line_text
                w_start = line_text.rfind(" ", 0, m.start()) + 1
                w_end = line_text.find(" ", m.end())
                if w_end == -1:
                    w_end = len(line_text)
                enclosing = line_text[w_start:w_end].strip()
                if _is_lookalike(enclosing):
                    continue
                line_candidates.append((m.start(), m.end(), token))

        if not line_candidates:
            continue

        # Build set of already-covered spans (from typed hits) for this chunk
        covered: List[Tuple[int, int]] = []
        if existing_hits:
            for eh in existing_hits:
                if eh.entity_type != "SUSPECT_OCR":
                    covered.append((eh.start, eh.end))

        # Isolate candidate spans — skip any span already covered by a typed hit
        if line_pos != -1:
            for s_start, s_end, tok in line_candidates:
                global_start = line_pos + s_start
                global_end = line_pos + s_end
                # Gap-only: skip if overlaps any typed hit
                if any(global_start < ce and global_end > cs for cs, ce in covered):
                    log.debug(
                        "SUSPECT_OCR skipped (covered by typed hit): %r [%d:%d]",
                        tok, global_start, global_end,
                    )
                    continue
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
                    gs, ge = m_tok.start(), m_tok.end()
                    if any(gs < ce and ge > cs for cs, ce in covered):
                        log.debug(
                            "SUSPECT_OCR (fallback) skipped (covered): %r", tok)
                        continue
                    hits.append(RawHit(
                        entity_type="SUSPECT_OCR",
                        start=gs,
                        end=ge,
                        text=tok,
                        confidence=0.85,
                        sensitivity="High",
                        method="ocr_low_confidence",
                    ))

    return hits
