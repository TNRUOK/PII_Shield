"""
detect/merge.py
---------------
Merge and de-duplicate all detection hits from rules, NER, and context layers.

Algorithm:
  1. Collect all RawHits for a chunk.
  2. Sort by (start, -confidence).
  3. Greedy overlap resolution: a new hit is merged with the current window
     if it overlaps. If same type → union spans. If different type → keep
     the one with higher confidence and record both methods.
  4. De-dup by (start, end, entity_type) keeping highest confidence.
  5. Assign sensitivity from config.sensitivity_map.
  6. Produce final Entity objects.

Returns a list of Entity objects.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from detect.rules import RawHit

log = logging.getLogger(__name__)


@dataclass
class Entity:
    """
    A fully resolved PII entity after merging all detection layers.
    This is the canonical output of the detection pipeline.
    """
    entity_type: str
    sensitivity: str
    confidence: float
    methods: List[str]        # e.g. ["rule", "ner", "context"]
    file: str
    location: str
    start: int
    end: int
    raw_value: str            # NEVER written to report output
    masked_value: str         # first + last char, stars in between
    placeholder: str = ""     # assigned later by redactor
    chunk_index: Optional[int] = None


def _mask(text: str) -> str:
    """
    Mask a PII value: keep first and last character, replace middle with stars.
    Examples: "Jane Smith" → "J********h", "j@example.com" → "j***********m"
    Min 3 chars: show first+last; 1-2 chars: all stars.
    """
    if len(text) <= 2:
        return "*" * len(text)
    return text[0] + "*" * (len(text) - 2) + text[-1]


def _sensitivity_for(entity_type: str, config: dict) -> str:
    """Look up sensitivity level from config.sensitivity_map."""
    s_map: Dict[str, List[str]] = config.get("sensitivity_map", {})
    for level, types in s_map.items():
        if entity_type in types:
            return level
    return "Medium"  # default


def _dedup_methods(methods: List[str]) -> List[str]:
    """Return unique method names preserving order."""
    seen = set()
    out = []
    for m in methods:
        for part in m.split("+"):
            if part not in seen:
                seen.add(part)
                out.append(part)
    return out


def merge_hits(
    hits: List[RawHit],
    file: str,
    location: str,
    config: dict,
) -> List[Entity]:
    """
    Merge, de-duplicate and resolve all raw hits for one chunk into Entity objects.

    Parameters
    ----------
    hits     : Combined hits from rules + NER + context layers.
    file     : Source file path.
    location : Chunk location string.
    config   : Loaded config dict.
    """
    if not hits:
        return []

    # Sort by start position, then descending confidence
    sorted_hits = sorted(hits, key=lambda h: (h.start, -h.confidence))

    merged: List[RawHit] = []
    current = sorted_hits[0]

    for hit in sorted_hits[1:]:
        # Check overlap: hit starts before current ends
        if hit.start < current.end:
            # Overlapping spans — always union (T5 fix: do not drop the tail of
            # a lower-confidence hit that extends beyond the higher-confidence one)
            union_start = min(current.start, hit.start)
            union_end   = max(current.end, hit.end)
            combined_method = current.method + "+" + hit.method

            if hit.entity_type == current.entity_type:
                # Same type → union spans, keep max confidence, combine methods
                current = RawHit(
                    entity_type=current.entity_type,
                    start=union_start,
                    end=union_end,
                    text="",  # will be re-derived from chunk text in Entity creation
                    confidence=max(current.confidence, hit.confidence),
                    sensitivity=current.sensitivity,
                    method=combined_method,
                )
            else:
                # Different type — always union spans, keep higher-confidence type
                if hit.confidence >= current.confidence:
                    winner = hit
                else:
                    winner = current
                current = RawHit(
                    entity_type=winner.entity_type,
                    start=union_start,
                    end=union_end,
                    text="",  # re-derived below
                    confidence=max(current.confidence, hit.confidence),
                    sensitivity=winner.sensitivity,
                    method=combined_method,
                )
        else:
            merged.append(current)
            current = hit

    merged.append(current)

    # Build Entity objects
    entities: List[Entity] = []
    for hit in merged:
        raw = hit.text  # May be empty if span-merged; caller should pass original text
        sensitivity = _sensitivity_for(hit.entity_type, config)
        methods = _dedup_methods([hit.method])

        entities.append(Entity(
            entity_type=hit.entity_type,
            sensitivity=sensitivity,
            confidence=round(hit.confidence, 4),
            methods=methods,
            file=file,
            location=location,
            start=hit.start,
            end=hit.end,
            raw_value=raw,
            masked_value=_mask(raw),
            placeholder="",  # filled in by redactor.py
        ))

    return entities


def detect_chunk(
    chunk,
    config: dict,
    *,
    use_ner: bool = True,
    use_context: bool = True,
    use_suspect: bool = True,
    use_repair: bool = True,
) -> List[Entity]:
    """
    Run detection layers on a single Chunk and return merged entities.
    Optional flags allow ablating individual stages (Item 5).
    """
    from detect.rules import run_rules
    from detect.ner import run_ner
    from detect.context import run_context
    from detect.suspect import find_suspect_ocr_hits
    from detect.ocr_repair import find_ocr_repair_hits

    if chunk.is_blocked() or not chunk.text.strip():
        return []

    is_ocr = (chunk.method == "ocr") or (getattr(chunk, "ocr_confidence", None) is not None)
    rule_hits = run_rules(chunk.text, config, is_ocr=is_ocr)

    all_hits = list(rule_hits)

    if use_ner:
        # Run NER per table cell (not on the pipe-joined row) to prevent pipe delimiter confusion
        if "table:" in chunk.location and "|" in chunk.text:
            ner_hits = []
            char_offset = 0
            for cell in chunk.text.split("|"):
                if cell.strip():
                    cell_hits = run_ner(cell, config)
                    for h in cell_hits:
                        h.start += char_offset
                        h.end += char_offset
                        ner_hits.append(h)
                char_offset += len(cell) + 1  # +1 for the pipe separator
        else:
            ner_hits = run_ner(chunk.text, config)
        all_hits.extend(ner_hits)

    if use_context:
        context_new, all_hits = run_context(
            chunk.location, chunk.text, all_hits, config)
        all_hits.extend(context_new)

    # Item 1: fail-closed redaction for low-confidence OCR lines
    if use_suspect:
        suspect_hits = find_suspect_ocr_hits(chunk, config)
        if suspect_hits:
            log.debug("SUSPECT_OCR: %d hits from low-confidence OCR lines in %s",
                      len(suspect_hits), chunk.location)
            all_hits.extend(suspect_hits)

    # Item 2: character-confusion normalisation for OCR candidate tokens
    if use_repair:
        repair_hits = find_ocr_repair_hits(chunk, config)
        if repair_hits:
            log.debug("OCR_REPAIR: %d normalised hits in %s",
                      len(repair_hits), chunk.location)
            all_hits.extend(repair_hits)

    # Re-derive raw_value from chunk text for span-merged hits
    entities = merge_hits(all_hits, chunk.file, chunk.location, config)
    for ent in entities:
        ent.chunk_index = getattr(chunk, "order_index", None)
        if not ent.raw_value:
            ent.raw_value = chunk.text[ent.start: ent.end]
            ent.masked_value = _mask(ent.raw_value)

    return entities

