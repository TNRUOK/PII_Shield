"""
detect/context.py
-----------------
Layer C: Context-aware confidence boosting.

Three sub-strategies:
  1. Table-aware column header boost
     If a chunk's location contains "table:" and "headers:" metadata,
     any hit whose span falls in a column whose header matches a PII keyword
     has its confidence multiplied by config.context.table_header_boost.

  2. Labelled-name patterns
     "Prepared by: Jane Smith", "Author: Bob Jones", etc.
     → emit PERSON hit at labelled_name_confidence.

  3. Unlabelled CAPITALISED strings in a "Name" column
     Two-to-four title-case words in a table column headed "name" or "employee"
     → PERSON hit at unlabelled_name_col_confidence.

run_context(chunk, existing_hits, config) -> additional_hits, boosted_hits
  Returns (new_hits, existing_hits_with_updated_confidence).
"""
from __future__ import annotations

import re
import logging
from typing import List, Tuple

from detect.rules import RawHit

log = logging.getLogger(__name__)

# Keywords that trigger column-header confidence boost
_HEADER_PII_KEYWORDS = re.compile(
    r"\b(name|employee|emp|contact|email|phone|mobile|owner|approver|staff|id|"
    r"user|users|member|members|reviewer|author|assignee|respondent|"
    r"peer\s+reviewer|qa\s+reviewer|client\s+contact|prepared\s+by|submitted\s+by)\b",
    re.IGNORECASE,
)

# Headers indicating a person/name column
_NAME_HEADER_KEYWORDS = re.compile(
    r"\b(name|employee|emp|staff|person|contact|owner|author|lead|manager|attendee|"
    r"user|users|member|members|reviewer|assignee|respondent|approver|"
    r"peer\s+reviewer|qa\s+reviewer|client\s+contact)\b",
    re.IGNORECASE,
)

# Words that — when present in a header — VETO any keyword match.
# An organisational-structure word beats a person-keyword match.
_HEADER_EXCLUSION_RE = re.compile(
    r"\b(group|groups|team|teams|department|departments|dept|role|roles|title|"
    r"permission|permissions|description|type|status|workflow|stage)\b",
    re.IGNORECASE,
)


def _header_is_name_column(header: str) -> bool:
    """Return True iff *header* should be treated as a person-name column.

    Rules (in order):
    1. If any exclusion word is present (word-boundary), return False.
    2. If a name-keyword is present (word-boundary), return True.
    """
    if _HEADER_EXCLUSION_RE.search(header):
        return False
    return bool(_NAME_HEADER_KEYWORDS.search(header))


def _header_is_pii_column(header: str) -> bool:
    """Return True iff *header* warrants a PII confidence boost.

    Same veto logic: exclusion words prevent an email/phone column in a
    workflow table (e.g. 'Group Status') from getting a spurious boost.
    """
    if _HEADER_EXCLUSION_RE.search(header):
        return False
    return bool(_HEADER_PII_KEYWORDS.search(header))

# Labelled keys for key-value formats
_LABEL_KEYS = (
    r"(?:prepared\s+by|approved\s+by|author(?:\(s\))?|owner|submitted\s+by|signed\s+by|"
    r"reviewer|peer\s+reviewer|qa\s+reviewer|client\s+contact|assignee|respondent|approver|"
    r"user|member|members|full\s+name|employee\s+name|staff\s+name)"
)

# Initials-name pattern: e.g. "TJ Harlan", "PQ Lindqvist", "J. Doe", "A. B. Smith", "J Doe", "RK Patel"
_INITIALS_NAME_PATTERN = r"[A-Z]{1,3}\.?\s+(?:[A-Z]\.?\s+)?[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?"
_TITLE_CASE_NAME_PATTERN = r"[A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,3}"

_NAME_OR_INITIALS_PATTERN = rf"(?:{_INITIALS_NAME_PATTERN}|{_TITLE_CASE_NAME_PATTERN})"

_INITIALS_NAME = re.compile(rf"\b({_INITIALS_NAME_PATTERN})\b")
_TITLE_CASE_NAME = re.compile(rf"\b({_TITLE_CASE_NAME_PATTERN})\b")

# Labelled-name pattern: keyword followed by name (allows optional newline for OCR output)
_LABELLED_NAME = re.compile(
    rf"(?i:\b{_LABEL_KEYS})\s*[:\-]\n?\s*({_NAME_OR_INITIALS_PATTERN})"
)

# Key-Value 2-column table label
_KV_LABEL_PAT = re.compile(
    rf"^(?:{_LABEL_KEYS})$",
    re.IGNORECASE,
)

_NAME_STOP_WORDS = {
    "employee", "staff", "id", "email", "phone", "mobile", "ssn", "dept",
    "department", "title", "role", "system", "product", "note", "approved",
    "prepared", "contact", "signatory", "owner", "author", "reviewed", "primary",
    "by", "section", "date", "status", "version", "questions", "user", "users",
    "member", "members", "reviewer", "assignee", "respondent", "custodian",
    "delegate", "stakeholder", "participant", "operator", "signoff", "manager",
    "lead", "attendee", "company", "services", "financial", "technologies",
    "solutions", "limited", "corp", "inc", "ltd"
}


def _parse_headers_from_location(location: str) -> List[str]:
    """Extract header texts embedded in the location string like [headers: A | B | C]."""
    m = re.search(r"\[headers:\s*([^\]]+)\]", location)
    if not m:
        return []
    return [h.strip().lower() for h in m.group(1).split("|")]


def _col_index_of_hit(row_text: str, start: int) -> int:
    """
    Determine which pipe-separated column a character offset falls in.
    Returns the 0-based column index.
    """
    col = 0
    for i, ch in enumerate(row_text):
        if i >= start:
            break
        if ch == "|":
            col += 1
    return col


def run_context(
    chunk_location: str,
    chunk_text: str,
    existing_hits: List[RawHit],
    config: dict,
) -> Tuple[List[RawHit], List[RawHit]]:
    """
    Apply context rules to a chunk.

    Parameters
    ----------
    chunk_location : The Chunk.location string.
    chunk_text     : The Chunk.text string.
    existing_hits  : Hits already detected by rules and NER layers.
    config         : Loaded config dict.

    Returns
    -------
    (new_hits, updated_existing_hits)
      new_hits             : Additional hits discovered by context rules.
      updated_existing_hits: The same list with confidence values possibly boosted.
    """
    ctx_cfg = config.get("context", {})
    boost = float(ctx_cfg.get("table_header_boost", 1.2))
    labelled_conf = float(ctx_cfg.get("labelled_name_confidence", 0.85))
    unlabelled_conf = float(ctx_cfg.get("unlabelled_name_col_confidence", 0.60))

    new_hits: List[RawHit] = []
    updated = list(existing_hits)  # copy so we can mutate confidences

    is_table_chunk = "table:" in chunk_location

    # --- Strategy 1: Table column header boost ---
    if is_table_chunk:
        headers = _parse_headers_from_location(chunk_location)
        # Determine which header columns are PII-relevant (exclusion veto applied)
        pii_col_indices = {
            i for i, h in enumerate(headers)
            if _header_is_pii_column(h)
        }

        for hit in updated:
            col_idx = _col_index_of_hit(chunk_text, hit.start)
            if col_idx in pii_col_indices:
                hit.confidence = min(hit.confidence * boost, 1.0)
                if "context" not in hit.method:
                    hit.method = hit.method + "+context"

        # --- Strategy 3: Unlabelled names in a name column (exclusion veto applied) ---
        name_col_indices = {
            i for i, h in enumerate(headers)
            if _header_is_name_column(h)
        }
        if name_col_indices:
            cells = chunk_text.split("|")
            char_offset = 0
            for col_i, cell in enumerate(cells):
                if col_i in name_col_indices:
                    # Match both full title-case names and initials-style names
                    cand_matches = (
                        list(_TITLE_CASE_NAME.finditer(cell)) +
                        list(_INITIALS_NAME.finditer(cell))
                    )
                    for m in cand_matches:
                        raw_val = m.group(1).strip()
                        words = raw_val.split()
                        if any(w.strip(":-.").lower() in _NAME_STOP_WORDS for w in words):
                            continue
                        global_start = char_offset + m.start(1)
                        global_end = char_offset + m.end(1)
                        covered = any(
                            h.start <= global_start and h.end >= global_end
                            for h in (updated + new_hits)
                        )
                        if not covered:
                            new_hits.append(RawHit(
                                entity_type="PERSON",
                                start=global_start,
                                end=global_end,
                                text=raw_val,
                                confidence=unlabelled_conf,
                                sensitivity="Medium",
                                method="context",
                            ))
                char_offset += len(cell) + 1  # +1 for the pipe separator

    # --- Strategy 2b: Two-column key-value tables (Key | Value) ---
    if "|" in chunk_text:
        cells = chunk_text.split("|")
        if len(cells) >= 2 and _KV_LABEL_PAT.match(cells[0].strip()):
            val_cell = cells[1]
            val_offset = len(cells[0]) + 1
            cand_matches = (
                list(_TITLE_CASE_NAME.finditer(val_cell)) +
                list(_INITIALS_NAME.finditer(val_cell))
            )
            for m in cand_matches:
                raw_val = m.group(1).strip()
                words = raw_val.split()
                if any(w.strip(":-.").lower() in _NAME_STOP_WORDS for w in words):
                    continue
                global_start = val_offset + m.start(1)
                global_end = val_offset + m.end(1)
                covered = any(
                    h.start <= global_start and h.end >= global_end
                    for h in (updated + new_hits)
                )
                if not covered:
                    new_hits.append(RawHit(
                        entity_type="PERSON",
                        start=global_start,
                        end=global_end,
                        text=raw_val,
                        confidence=labelled_conf,
                        sensitivity="Medium",
                        method="context",
                    ))

    # --- Strategy 2: Labelled-name patterns (Key: Value) ---
    for m in _LABELLED_NAME.finditer(chunk_text):
        raw_name = m.group(1)
        words = raw_name.split()
        valid_words = []
        for w in words:
            clean_w = w.strip(":-.").lower()
            if clean_w in _NAME_STOP_WORDS:
                break
            valid_words.append(w)

        if not valid_words:
            continue

        name_text = " ".join(valid_words)
        name_start = m.start(1)
        name_end = name_start + len(name_text)

        # Don't add if a PERSON hit already covers this span
        covered = any(
            h.entity_type == "PERSON" and
            h.start <= name_start and h.end >= name_end
            for h in (updated + new_hits)
        )
        if not covered:
            new_hits.append(RawHit(
                entity_type="PERSON",
                start=name_start,
                end=name_end,
                text=name_text,
                confidence=labelled_conf,
                sensitivity="Medium",
                method="context",
            ))

    return new_hits, updated
