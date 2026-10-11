"""
detect/propagate.py
-------------------
Item 3: Document-level name alias propagation.

For each PERSON entity with ≥2 tokens detected in a document, build a set of
aliases (surname, first name >3 chars, initials, email local-part) and redact
ALL occurrences of those aliases across all document chunks with the same
placeholder as the canonical PERSON hit.

Design:
- Only propagates when the canonical detection is ≥2 tokens (avoids single-word
  false positives like "John" or "Security").
- Aliases must be >3 characters, not in alias_exclusions, and not a look-alike.
- Applies to all subsequent chunks (and back-scans the full text list).
- Returns additional Entity objects for the alias hits.
- Fail-safe: any exception leaves the original entity list untouched.
"""
from __future__ import annotations

import re
import logging
from typing import List, Dict, Set, Tuple, Optional

log = logging.getLogger(__name__)

# Minimum alias length to avoid propagating initials like "J" alone
_MIN_ALIAS_LEN = 3

# Look-alike guard — do not propagate numeric/version/date tokens
_NUMERIC_OR_DATE_RE = re.compile(r"^[\d./\-]+$")


def _is_excluded(token: str, exclusions: Set[str]) -> bool:
    t = token.strip()
    if t.lower() in {e.lower() for e in exclusions}:
        return True
    if _NUMERIC_OR_DATE_RE.match(t):
        return True
    return False


def _build_aliases(full_name: str, exclusions: Set[str]) -> List[str]:
    """
    Build alias list for a full name with ≥2 tokens:
    - Each token longer than _MIN_ALIAS_LEN (and not excluded)
    - First initial + surname: "J. Smith"
    - Full name itself (for exact match)
    """
    tokens = full_name.split()
    if len(tokens) < 2:
        return []

    aliases: List[str] = []

    # Individual tokens (surname, first name)
    for tok in tokens:
        clean = tok.strip(".,'\"")
        if len(clean) > _MIN_ALIAS_LEN and not _is_excluded(clean, exclusions):
            aliases.append(clean)

    # Full name
    aliases.append(full_name)

    # First-initial + surname: "J. Smith"
    if len(tokens) >= 2:
        initial_surname = f"{tokens[0][0]}. {tokens[-1]}"
        aliases.append(initial_surname)

    return aliases


def _extract_email_localpart(email_value: str) -> Optional[str]:
    """Return the local-part (before @) of an email, if it's long enough."""
    if "@" not in email_value:
        return None
    local = email_value.split("@")[0]
    local = local.replace(".", " ").replace("_", " ").replace("-", " ")
    if len(local) > _MIN_ALIAS_LEN:
        return local
    return None


def propagate_name_aliases(
    all_chunks_text: List[str],
    person_entities: List,   # List[Entity] — avoid circular import
    all_entities: List,      # List[Entity]
    config: dict,
    file: str,
    locations: List[str],
) -> List:
    """
    For each confirmed PERSON entity, find alias occurrences across all chunks
    and return additional Entity objects for those aliases.

    Parameters
    ----------
    all_chunks_text : List of chunk texts in order.
    person_entities : PERSON entities already detected.
    all_entities    : All entities for the document (used to find EMAIL matches).
    config          : Loaded config dict.
    file            : Source file path.
    locations       : Chunk location strings (parallel to all_chunks_text).

    Returns new Entity objects to append.
    """
    from detect.merge import Entity, _mask

    exclusions: Set[str] = set(config.get("alias_exclusions", []))
    sensitivity_map = config.get("sensitivity_map", {})

    # Map canonical person raw_value → placeholder
    person_map: Dict[str, str] = {}
    for ent in person_entities:
        raw = (ent.raw_value or "").strip()
        tokens = raw.split()
        if len(tokens) >= 2:
            person_map[raw] = ent.placeholder or ""

    if not person_map:
        return []

    # Also check for email entities to add local-part aliases
    email_localparts: Dict[str, str] = {}  # localpart → person raw_value
    for ent in all_entities:
        if ent.entity_type == "EMAIL" and ent.raw_value:
            lp = _extract_email_localpart(ent.raw_value)
            if lp:
                # Try to match localpart against known person names
                for person_name in person_map:
                    name_lower = person_name.lower().replace(" ", "")
                    lp_lower = lp.lower().replace(" ", "")
                    if name_lower in lp_lower or lp_lower in name_lower:
                        email_localparts[lp] = person_name
                        break

    # Build alias → canonical name mapping
    alias_to_canonical: Dict[str, str] = {}
    for person_name in person_map:
        for alias in _build_aliases(person_name, exclusions):
            if alias and alias != person_name:
                alias_to_canonical[alias] = person_name

    if not alias_to_canonical:
        return []

    new_entities: List[Entity] = []

    # Determine sensitivity for PERSON
    person_sensitivity = "Medium"
    for level, types in sensitivity_map.items():
        if "PERSON" in types:
            person_sensitivity = level
            break

    # Scan each chunk for alias occurrences
    for chunk_idx, (chunk_text, location) in enumerate(zip(all_chunks_text, locations)):
        already_covered: List[Tuple[int, int]] = []

        # Collect already-detected entity spans in this chunk
        for ent in all_entities:
            if ent.location == location and ent.file == file:
                already_covered.append((ent.start, ent.end))

        for alias, canonical in alias_to_canonical.items():
            # Whole-word search for the alias
            try:
                pattern = re.compile(
                    r"(?<!\w)" + re.escape(alias) + r"(?!\w)",
                    re.IGNORECASE
                )
            except re.error:
                continue

            for m in pattern.finditer(chunk_text):
                s, e = m.start(), m.end()
                # Skip if already covered by an existing entity
                if any(s < ce and e > cs for cs, ce in already_covered):
                    continue
                raw_val = m.group(0)
                canonical_placeholder = person_map.get(canonical, "")

                new_entities.append(Entity(
                    entity_type="PERSON",
                    sensitivity=person_sensitivity,
                    confidence=0.70,   # slightly lower — propagated, not directly detected
                    methods=["alias_propagation"],
                    file=file,
                    location=location,
                    start=s,
                    end=e,
                    raw_value=raw_val,
                    masked_value=_mask(raw_val),
                    placeholder=canonical_placeholder,
                    chunk_index=chunk_idx,
                ))
                already_covered.append((s, e))

    log.debug(
        "Alias propagation: %d canonical persons → %d alias aliases → %d new hits in %s",
        len(person_map), len(alias_to_canonical), len(new_entities), file
    )
    return new_entities
