"""
verify/leakage.py
-----------------
Leakage check: re-run the full detection pipeline on already-redacted chunks,
with known-value propagation and independent regex verification.

Any detection hit that is NOT a placeholder token ([TYPE_N] pattern) is a LEAK.
A leak causes the file to be marked downstream_safe=False.

This is the final safety gate before content is cleared for LLM use.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional, Union, Dict, Set, Any

from extract.models import Chunk
from detect.merge import detect_chunk, Entity, _mask

log = logging.getLogger(__name__)

# Matches our placeholder format [TYPE_N] — used to exclude them from re-detection
_PLACEHOLDER_RE = re.compile(r"\[[A-Z_]+_\d+\]")

# Independent regex patterns for second pass / unverified residue / partial leaks
_PARTIAL_SSN_RE = re.compile(
    r"(?<!\d)\d{3}-\d{2}-(?:[0-9Xx*#]{1,4}|(?=\s*██))"  # e.g. 123-45-XXXX, 123-45-67, 123-45- adjacent to stripped placeholder
    r"|██\s*(?:-?\d{2}-\d{4})(?!\d)"                   # e.g. adjacent to stripped placeholder -45-6789
    r"|(?<!\d)\d{3}-\d{2}-\d{1,4}(?!\d)"                # e.g. 123-45-67
)
_RESIDUE_INITIALS_RE = re.compile(r"\b([A-Z]{1,3}\s+[A-Z][a-z]{2,})\b")
_RESIDUE_CAP_NGRAMS_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,2})\b")
_RESIDUE_LONG_DIGITS_RE = re.compile(r"\b\d{8,}\b")


def _strip_placeholders(text: str) -> str:
    """
    Replace placeholder tokens with a neutral block symbol so NER tokenizers
    cannot mistakenly bridge across redacted spans or trigger false positives.
    Uses '██' rather than ', ' to avoid creating comma patterns that could
    falsely match date/phone/IP regexes.
    """
    return _PLACEHOLDER_RE.sub(" ██ ", text)


@dataclass
class LeakResult:
    """Result of the leakage check for one file."""
    file: str
    downstream_safe: bool
    leaks: List[Entity]  # any remaining PII hits after redaction
    confirmed_leaks: List[Entity] = field(default_factory=list)
    unverified_residue: List[Dict[str, Any]] = field(default_factory=list)
    residue: List[str] = field(default_factory=list)  # masked leaked / residue strings (NO raw PII)

    def __post_init__(self):
        if not self.confirmed_leaks and self.leaks:
            self.confirmed_leaks = list(self.leaks)
        if not self.residue and self.leaks:
            self.residue = [_mask(getattr(e, "raw_value", "") or getattr(e, "masked_value", "")) for e in self.leaks]

    def summary(self) -> str:
        if self.downstream_safe:
            msg = f"✅ {self.file}: downstream_safe=YES (0 leaks)"
            if self.unverified_residue:
                msg += f" (residue items: {len(self.unverified_residue)})"
            return msg
        return (
            f"❌ {self.file}: downstream_safe=NO — "
            f"{len(self.leaks)} leak(s): "
            + ", ".join(f"{e.entity_type}@{e.location}" for e in self.leaks[:5])
        )


def _extract_known_values(known_values: Any) -> tuple[Dict[str, str], Set[str]]:
    """
    Extract normalized (casefolded) known raw PII strings and type mappings.
    Returns (raw_to_type, set_of_raw_values).
    """
    raw_to_type: Dict[str, str] = {}
    known_set: Set[str] = set()

    if known_values is None:
        return raw_to_type, known_set

    # PlaceholderMapping support
    if hasattr(known_values, "_placeholder_to_raw"):
        for ph, raw in known_values._placeholder_to_raw.items():
            if raw and isinstance(raw, str):
                s = raw.strip()
                if len(s) >= 2 and not _PLACEHOLDER_RE.fullmatch(s):
                    etype = ph.strip("[]").rsplit("_", 1)[0]
                    raw_to_type[s.lower()] = etype
                    known_set.add(s)
        return raw_to_type, known_set

    if isinstance(known_values, dict):
        for k, v in known_values.items():
            if isinstance(v, str):
                s = v.strip()
                if len(s) >= 2 and not _PLACEHOLDER_RE.fullmatch(s):
                    if isinstance(k, str) and k.startswith("["):
                        etype = k.strip("[]").rsplit("_", 1)[0]
                        raw_to_type[s.lower()] = etype
                    known_set.add(s)
            if isinstance(k, str):
                s = k.strip()
                if len(s) >= 2 and not _PLACEHOLDER_RE.fullmatch(s) and not s.startswith("["):
                    known_set.add(s)
        return raw_to_type, known_set

    if isinstance(known_values, (list, tuple, set)):
        for item in known_values:
            if isinstance(item, str):
                s = item.strip()
                if len(s) >= 2 and not _PLACEHOLDER_RE.fullmatch(s):
                    known_set.add(s)

    return raw_to_type, known_set


def check_leakage(
    redacted_chunks: List[Chunk],
    config: dict,
    known_values: Optional[Union[List[str], Dict[str, Any], Any]] = None,
) -> LeakResult:
    """
    Run detection on redacted chunks. Any hit outside a placeholder is a leak.

    Parameters
    ----------
    redacted_chunks : Chunks after redaction has been applied.
    config          : Loaded config dict.
    known_values    : Optional raw PII values (or PlaceholderMapping) for propagation.

    Returns
    -------
    LeakResult with downstream_safe flag, confirmed leaks, unverified residue, and masked residue.
    """
    if not redacted_chunks:
        return LeakResult(file="", downstream_safe=True, leaks=[], confirmed_leaks=[], unverified_residue=[], residue=[])

    file_path = redacted_chunks[0].file
    leaks: List[Entity] = []
    unverified_residue: List[Dict[str, Any]] = []
    residue_set: List[str] = []
    chunks_scanned = 0

    raw_to_type, known_set = _extract_known_values(known_values)

    for chunk in redacted_chunks:
        if chunk.is_blocked():
            continue  # blocked chunks have no text — nothing to scan

        chunks_scanned += 1
        # Strip placeholders before detection to avoid false alarm on placeholder text
        scan_text = _strip_placeholders(chunk.text)

        # 1. Primary detection scan on redacted text (rules + NER + context)
        scan_chunk = SimpleNamespace(
            file=chunk.file,
            location=chunk.location,
            text=scan_text,
            method=chunk.method,
            ocr_confidence=chunk.ocr_confidence,
            order_index=chunk.order_index,
            blocked_reason=None,
            is_blocked=lambda: False,
        )

        entities = detect_chunk(scan_chunk, config)
        existing_spans: Set[tuple[int, int]] = set()

        if entities:
            for ent in entities:
                log.warning(
                    "LEAK detected in %s @ %s: %s (conf=%.2f)",
                    file_path, chunk.location, ent.entity_type, ent.confidence
                )
                if not getattr(ent, "masked_value", None):
                    ent.masked_value = _mask(ent.raw_value)
                leaks.append(ent)
                existing_spans.add((ent.start, ent.end))
                if ent.masked_value not in residue_set:
                    residue_set.append(ent.masked_value)

        # 2. Known-value propagation: exact case-insensitive search for known raw PII
        for known_val in known_set:
            val_lower = known_val.lower()
            ent_type = raw_to_type.get(val_lower, "KNOWN_PII")
            # If single-word PERSON name, require word boundary and title/exact case
            # to prevent matching common nouns (e.g. 'glass', 'page', 'may') in other chunks
            if ent_type == "PERSON" and len(known_val.split()) == 1:
                pattern = re.compile(rf"\b{re.escape(known_val)}\b")
            elif any(c.isalnum() for c in known_val):
                pattern = re.compile(rf"\b{re.escape(known_val)}\b", re.IGNORECASE)
            else:
                pattern = re.compile(re.escape(known_val), re.IGNORECASE)

            for m in pattern.finditer(scan_text):
                m_span = (m.start(), m.end())
                # Check if already covered by an existing leak entity
                if not any(s <= m.start() and m.end() <= e for (s, e) in existing_spans):
                    matched_raw = scan_text[m.start():m.end()]
                    masked = _mask(matched_raw)
                    ent_type = raw_to_type.get(val_lower, "KNOWN_PII")
                    leak_ent = Entity(
                        entity_type=ent_type,
                        start=m.start(),
                        end=m.end(),
                        file=chunk.file,
                        location=chunk.location,
                        confidence=1.0,
                        sensitivity="High",
                        methods=["known_value_verifier"],
                        raw_value=matched_raw,
                        masked_value=masked,
                    )
                    log.warning(
                        "KNOWN-VALUE LEAK in %s @ %s: %s (%s)",
                        file_path, chunk.location, ent_type, masked
                    )
                    leaks.append(leak_ent)
                    existing_spans.add(m_span)
                    if masked not in residue_set:
                        residue_set.append(masked)

            # Check for partial SSN prefix/suffix of known SSNs
            if re.match(r"^\d{3}-\d{2}-\d{4}$", known_val):
                prefix = known_val[:7]  # e.g. 123-45-
                if prefix in scan_text:
                    for m in re.finditer(re.escape(prefix), scan_text):
                        if not any(s <= m.start() and m.end() <= e for (s, e) in existing_spans):
                            matched_raw = scan_text[m.start():m.end()]
                            masked = _mask(matched_raw)
                            leak_ent = Entity(
                                entity_type="SSN",
                                start=m.start(),
                                end=m.end(),
                                file=chunk.file,
                                location=chunk.location,
                                confidence=1.0,
                                sensitivity="High",
                                methods=["known_value_partial_ssn"],
                                raw_value=matched_raw,
                                masked_value=masked,
                            )
                            log.warning("KNOWN-VALUE PARTIAL SSN LEAK in %s @ %s: %s", file_path, chunk.location, masked)
                            leaks.append(leak_ent)
                            existing_spans.add((m.start(), m.end()))
                            if masked not in residue_set:
                                residue_set.append(masked)

        # 3. Partial SSN pattern check (independent of known values)
        for m in _PARTIAL_SSN_RE.finditer(scan_text):
            if not any(s <= m.start() and m.end() <= e for (s, e) in existing_spans):
                matched_raw = scan_text[m.start():m.end()].replace("██", "").strip()
                if matched_raw and len(matched_raw) >= 3:
                    masked = _mask(matched_raw)
                    leak_ent = Entity(
                        entity_type="SSN",
                        start=m.start(),
                        end=m.end(),
                        file=chunk.file,
                        location=chunk.location,
                        confidence=0.95,
                        sensitivity="High",
                        methods=["partial_ssn_verifier"],
                        raw_value=matched_raw,
                        masked_value=masked,
                    )
                    log.warning("PARTIAL SSN LEAK in %s @ %s: %s", file_path, chunk.location, masked)
                    leaks.append(leak_ent)
                    existing_spans.add((m.start(), m.end()))
                    if masked not in residue_set:
                        residue_set.append(masked)

        # 4. Independent Second Pass: check for unverified residue (initials, long digits)
        # Unverified residue is recorded for audit and masked in residue; does not fail downstream_safe
        for m in _RESIDUE_INITIALS_RE.finditer(scan_text):
            if not any(s <= m.start() and m.end() <= e for (s, e) in existing_spans):
                val = m.group(0).strip()
                masked = _mask(val)
                unverified_residue.append({
                    "location": chunk.location,
                    "type": "INITIALS_RESIDUE",
                    "masked": masked,
                })
                if masked not in residue_set:
                    residue_set.append(masked)

        for m in _RESIDUE_LONG_DIGITS_RE.finditer(scan_text):
            if not any(s <= m.start() and m.end() <= e for (s, e) in existing_spans):
                val = m.group(0).strip()
                masked = _mask(val)
                unverified_residue.append({
                    "location": chunk.location,
                    "type": "LONG_DIGITS_RESIDUE",
                    "masked": masked,
                })
                if masked not in residue_set:
                    residue_set.append(masked)

    log.info(
        "Leakage scan complete: %d chunks scanned, %d confirmed leaks, %d residue items in %s",
        chunks_scanned, len(leaks), len(unverified_residue), file_path,
    )

    result = LeakResult(
        file=file_path,
        downstream_safe=len(leaks) == 0,
        leaks=leaks,
        confirmed_leaks=leaks,
        unverified_residue=unverified_residue,
        residue=residue_set,
    )
    log.info(result.summary())
    return result
