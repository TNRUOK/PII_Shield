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
    r"\b(name|employee|emp|contact|email|phone|mobile|owner|approver|staff|id)\b",
    re.IGNORECASE,
)

# Labelled-name pattern: keyword followed by name
_LABELLED_NAME = re.compile(
    r"(?i:\b(?:prepared\s+by|approved\s+by|author|owner|submitted\s+by|signed\s+by))\s*[:\-]\s*"
    r"([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,3})"
)

# Title-case name: 2-4 capitalised words (used in column context)
_TITLE_CASE_NAME = re.compile(
    r"\b([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,3})\b"
)


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
        # Determine which header columns are PII-relevant
        pii_col_indices = {
            i for i, h in enumerate(headers)
            if _HEADER_PII_KEYWORDS.search(h)
        }

        for hit in updated:
            col_idx = _col_index_of_hit(chunk_text, hit.start)
            if col_idx in pii_col_indices:
                hit.confidence = min(hit.confidence * boost, 1.0)
                if "context" not in hit.method:
                    hit.method = hit.method + "+context"

        # Strategy 3: unlabelled capitalised strings in a "name" column
        name_col_indices = {
            i for i, h in enumerate(headers)
            if re.search(r"\bname\b|\bemployee\b", h, re.IGNORECASE)
        }
        if name_col_indices:
            cells = chunk_text.split("|")
            char_offset = 0
            for col_i, cell in enumerate(cells):
                if col_i in name_col_indices:
                    m = _TITLE_CASE_NAME.search(cell)
                    if m:
                        global_start = char_offset + m.start()
                        global_end = char_offset + m.end()
                        # Only add if not already covered by existing hits
                        covered = any(
                            h.start <= global_start and h.end >= global_end
                            for h in updated
                        )
                        if not covered:
                            new_hits.append(RawHit(
                                entity_type="PERSON",
                                start=global_start,
                                end=global_end,
                                text=m.group(1),
                                confidence=unlabelled_conf,
                                sensitivity="Medium",
                                method="context",
                            ))
                char_offset += len(cell) + 1  # +1 for the pipe separator

    # --- Strategy 2: Labelled-name patterns ---
    for m in _LABELLED_NAME.finditer(chunk_text):
        name_text = m.group(1)
        name_start = m.start(1)
        name_end = m.end(1)
        # Don't add if a PERSON hit already covers this span
        covered = any(
            h.entity_type == "PERSON" and
            h.start <= name_start and h.end >= name_end
            for h in updated
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
