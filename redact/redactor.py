"""
redact/redactor.py
------------------
Apply placeholder redaction to Chunk text while preserving document structure.

Algorithm:
  - For each chunk, collect all entities whose location matches.
  - Sort entities by start offset in DESCENDING order so that replacing
    later offsets first does not shift earlier offsets.
  - Replace text[start:end] with the placeholder token.
  - Table pipe separators and whitespace padding are preserved because
    we only replace within the matched span.
  - Populate Entity.placeholder for every entity processed.

redact_chunks(chunks, entities_by_location, mapping, config) -> redacted_chunks
"""
from __future__ import annotations

import copy
import logging
from typing import List, Dict

from extract.models import Chunk
from detect.merge import Entity
from redact.mapping import PlaceholderMapping

log = logging.getLogger(__name__)


def _apply_redactions(text: str, entities: List[Entity], mapping: PlaceholderMapping) -> str:
    """
    Replace all entity spans in `text` with placeholder tokens.
    Entities must refer to offsets within this specific text.
    Processing in reverse order preserves offset validity.
    """
    # Sort descending by start so we replace right-to-left
    sorted_ents = sorted(entities, key=lambda e: e.start, reverse=True)
    chars = list(text)  # mutable list for efficient in-place replacement

    for ent in sorted_ents:
        placeholder = mapping.get_or_create(ent.entity_type, ent.raw_value)
        ent.placeholder = placeholder

        # Safety check: ensure offsets are within bounds
        if ent.start < 0 or ent.end > len(chars) or ent.start >= ent.end:
            log.warning(
                "Skipping out-of-bounds entity %s at [%d:%d] (text len %d)",
                ent.entity_type, ent.start, ent.end, len(chars)
            )
            continue

        chars[ent.start: ent.end] = list(placeholder)

    return "".join(chars)


def redact_chunks(
    chunks: List[Chunk],
    all_entities: List[Entity],
    mapping: PlaceholderMapping,
    config: dict,
) -> List[Chunk]:
    """
    Produce a list of redacted Chunk copies.

    Parameters
    ----------
    chunks       : Original extracted chunks.
    all_entities : All Entity objects (from all chunks of this file).
    mapping      : PlaceholderMapping for this document.
    config       : Loaded config dict.

    Returns
    -------
    New list of Chunk objects with redacted text. Blocked chunks pass through
    unchanged (they have no text to redact).
    """
    # Build lookups: chunk_index if present, fallback to location
    ents_by_chunk_idx: Dict[int, List[Entity]] = {}
    ents_by_location: Dict[str, List[Entity]] = {}
    for ent in all_entities:
        if getattr(ent, "chunk_index", None) is not None:
            ents_by_chunk_idx.setdefault(ent.chunk_index, []).append(ent)
        else:
            ents_by_location.setdefault(ent.location, []).append(ent)

    redacted_chunks: List[Chunk] = []

    for chunk in chunks:
        if chunk.is_blocked():
            # BLOCKED chunks pass through — no text to redact
            redacted_chunks.append(copy.copy(chunk))
            continue

        if getattr(chunk, "order_index", None) in ents_by_chunk_idx:
            chunk_entities = ents_by_chunk_idx[chunk.order_index]
        else:
            chunk_entities = ents_by_location.get(chunk.location, [])
        if not chunk_entities:
            redacted_chunks.append(copy.copy(chunk))
            continue

        redacted_text = _apply_redactions(chunk.text, chunk_entities, mapping)

        new_chunk = copy.copy(chunk)
        new_chunk.text = redacted_text
        redacted_chunks.append(new_chunk)

    return redacted_chunks
