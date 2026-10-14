"""
detect/propagate.py
-------------------
Item 3 & Issue 6: Document-level name alias propagation.

For each PERSON entity with ≥2 tokens detected in a document, and for each
detected EMAIL address (deriving surnames from local parts):
- Surnames (length >= 4, capitalized) propagate document-wide.
- First initial + surname ("X. Surname") forms propagate document-wide.
- First names propagate only when the full name was detected in the document
  and the first name is not in alias_exclusions or common-word stoplist.
- Retains conservative gating: sources containing non_person_terms, alias_source_stoplist,
  or person_denylist are never propagated ("Menon Road", "Security Group", etc. protected).
- Whole-word, case-sensitive matching.
"""
from __future__ import annotations

import re
import logging
from typing import List, Dict, Set, Tuple, Optional

log = logging.getLogger(__name__)

# Minimum alias length to avoid propagating short tokens
_MIN_ALIAS_LEN = 4

# Look-alike guard — do not propagate numeric/version/date tokens
_NUMERIC_OR_DATE_RE = re.compile(r"^[\d./\-]+$")

# Common English words that might match first names but should not propagate alone
_COMMON_WORD_STOPLIST = {
    "young", "brown", "white", "black", "green", "may", "will", "rose", "grace",
    "hope", "faith", "dawn", "robin", "cliff", "bill", "mark", "grant", "dean",
    "chase", "miles", "lane", "gene", "page", "art", "guy", "major", "general",
    "august", "april", "june", "july", "amber", "crystal", "penny", "berry",
    "glen", "dale", "wade", "reed", "joy", "gloria", "holly", "ivy", "summer"
}


def _is_excluded(token: str, exclusions: Set[str]) -> bool:
    t = token.strip()
    if t.lower() in {e.lower() for e in exclusions}:
        return True
    if _NUMERIC_OR_DATE_RE.match(t):
        return True
    return False


def _build_aliases_for_name(full_name: str, exclusions: Set[str]) -> List[str]:
    """
    Build alias list for a full name with ≥2 tokens:
    - Surname (last token, len >= 4)
    - First initial + surname: "J. Smith"
    - First name (first token, len >= 4): only if not in common words
    - Full name itself (for exact match)
    """
    tokens = full_name.split()
    if len(tokens) < 2:
        return []

    aliases: List[str] = []

    # Surname (last token)
    surname = tokens[-1].strip(".,'\"")
    if len(surname) >= _MIN_ALIAS_LEN and not _is_excluded(surname, exclusions):
        aliases.append(surname)

    # First initial + surname: "M. Achebe"
    initial = tokens[0][0].upper()
    initial_surname = f"{initial}. {surname}"
    aliases.append(initial_surname)

    # First name (only if not in common words or exclusions)
    first_name = tokens[0].strip(".,'\"")
    if (len(first_name) >= _MIN_ALIAS_LEN and
            not _is_excluded(first_name, exclusions) and
            first_name.lower() not in _COMMON_WORD_STOPLIST):
        aliases.append(first_name)

    # Full name
    aliases.append(full_name)

    return aliases


def _derive_email_surnames(email: str, veto_words: Set[str]) -> List[Tuple[str, str, str]]:
    """
    Derive (surname, initial_surname, full_canonical) from email local part.
    e.g. 'm.achebe@cadencecorp.net' -> ('Achebe', 'M. Achebe', 'M. Achebe')
    """
    if "@" not in email:
        return []
    local = email.split("@")[0].strip()
    results = []

    # Case 1: Dot / underscore / hyphen separated (m.achebe, t.lindqvist, p.raghavan, a.sharma72)
    parts = [p for p in re.split(r"[._\-]", local) if p]
    if len(parts) >= 2:
        surname_raw = re.sub(r"\d+$", "", parts[-1])
        first_raw = re.sub(r"\d+$", "", parts[0])
        if len(surname_raw) >= _MIN_ALIAS_LEN and surname_raw.lower() not in veto_words:
            surname = surname_raw.capitalize()
            initial = first_raw[0].upper() + "." if first_raw else ""
            init_surname = f"{initial} {surname}".strip()
            results.append((surname, init_surname, init_surname))
        if (len(first_raw) >= _MIN_ALIAS_LEN and
                first_raw.lower() not in veto_words and
                first_raw.lower() not in _COMMON_WORD_STOPLIST):
            first_name = first_raw.capitalize()
            results.append((first_name, first_name, first_name))
    elif len(local) >= 5:
        # Case 2: Concatenated 1-letter prefix + surname (e.g. mvance -> Vance, erostova -> Rostova)
        clean_local = re.sub(r"\d+$", "", local)
        if len(clean_local) >= 5 and clean_local.isalpha():
            init = clean_local[0].upper() + "."
            surname_raw = clean_local[1:]
            if len(surname_raw) >= _MIN_ALIAS_LEN and surname_raw.lower() not in veto_words:
                surname = surname_raw.capitalize()
                init_surname = f"{init} {surname}"
                results.append((surname, init_surname, init_surname))

    return results


def propagate_name_aliases(
    all_chunks_text: List[str],
    person_entities: List,   # List[Entity]
    all_entities: List,      # List[Entity]
    config: dict,
    file: str,
    locations: List[str],
) -> List:
    """
    For each confirmed PERSON entity and detected EMAIL, find alias occurrences
    across all chunks and return additional Entity objects for those aliases.
    """
    from detect.merge import Entity, _mask

    if not config.get("alias_propagation", True):
        return []

    exclusions: Set[str] = set(config.get("alias_exclusions", []))
    source_stoplist: Set[str] = {
        w.lower() for w in config.get(
            "alias_source_stoplist",
            [
                "Road", "Street", "Avenue", "Lane", "Drive", "Park",
                "Square", "Bank", "Group", "Team", "Board", "Office",
                "Center", "Centre", "Tower", "House", "Bridge", "City",
                "Valley", "Hill", "Hall", "Court", "Plaza", "Market",
                "Systems", "Services", "Solutions", "Partners", "Holdings",
                "Logistics", "Review", "Release", "Quality",
            ]
        )
    }
    non_person_terms: Set[str] = {
        w.lower() for w in config.get(
            "non_person_terms",
            [
                "Committee", "Office", "Group", "Board", "Risk", "Privacy",
                "Security", "Compliance", "Audit", "Technology", "Operations"
            ]
        )
    }
    denylist: Set[str] = {w.lower() for w in config.get("person_denylist", [])}
    veto_words = source_stoplist | non_person_terms | denylist | {e.lower() for e in exclusions}

    max_per_doc: int = int(config.get("alias_max_per_document", 50))
    sensitivity_map = config.get("sensitivity_map", {})

    # Map canonical person raw_value → placeholder
    person_map: Dict[str, str] = {}
    alias_to_canonical: Dict[str, str] = {}

    for ent in person_entities:
        raw = (ent.raw_value or "").strip()
        tokens = raw.split()
        if len(tokens) < 2:
            continue
        # Stoplist gating: reject if any token is in veto words
        if any(tok.lower().rstrip(".,;'") in veto_words for tok in tokens):
            log.debug("Alias propagation: skipping vetoed source %r", raw)
            continue

        placeholder = ent.placeholder or ""
        person_map[raw] = placeholder

        # Build aliases for confirmed person name
        for alias in _build_aliases_for_name(raw, exclusions):
            if alias and alias != raw:
                alias_to_canonical[alias] = raw

    # Issue 6b: Email-derived surnames
    for ent in all_entities:
        if ent.entity_type == "EMAIL" and ent.raw_value:
            for surname, init_surname, canonical in _derive_email_surnames(ent.raw_value, veto_words):
                if surname not in alias_to_canonical:
                    # Look if surname matches an existing person
                    matched_person = next(
                        (p for p in person_map if p.endswith(surname) or surname in p),
                        None
                    )
                    if matched_person:
                        alias_to_canonical[surname] = matched_person
                        alias_to_canonical[init_surname] = matched_person
                    else:
                        person_map[canonical] = ent.placeholder or ""
                        alias_to_canonical[surname] = canonical
                        alias_to_canonical[init_surname] = canonical

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

        # Sort aliases by length descending so longer aliases ("M. Achebe") match before ("Achebe")
        sorted_aliases = sorted(alias_to_canonical.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            canonical = alias_to_canonical[alias]
            # Whole-word, case-sensitive search for the alias
            try:
                pattern = re.compile(
                    r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])"
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
                    confidence=0.80,   # alias propagation
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
                if len(new_entities) >= max_per_doc:
                    log.debug("Alias propagation cap reached (%d) for %s", max_per_doc, file)
                    return new_entities

    log.debug(
        "Alias propagation: %d canonical persons → %d aliases → %d new hits in %s",
        len(person_map), len(alias_to_canonical), len(new_entities), file
    )
    return new_entities
