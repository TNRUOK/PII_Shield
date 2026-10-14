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
from detect.ner import _FORM_LABEL_DENYLIST, _FORM_LABEL_WORDS

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
_RESIDUE_CAP_NGRAMS_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:[^\S\r\n]+[A-Z][a-z]{2,}){1,2})\b")
_RESIDUE_LONG_DIGITS_RE = re.compile(r"\b\d{8,}\b")

# Phase 6 independent leak scanners (residual PII patterns)
_9XX_SSN_RE = re.compile(r"(?<!\d)9\d{2}-\d{2}-\d{4}(?!\d)")
_INTL_PHONE_RE = re.compile(r"\+\d{1,4}[\s.-]?(?:\(?\d{1,5}\)?[\s.-]?)?\d{2,4}[\s.-]?\d{3,4}(?:[\s.-]?\d{3,4})?\b")
_RESIDUE_DOB_RE = re.compile(
    r"\b(?:DOB|Date\s+of\s+Birth|Born)\s*[:\-]?\s*(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",
    re.IGNORECASE,
)
_RESIDUE_NARRATIVE_ADDR_RE = re.compile(
    r"\b(?:lives\s+at|resides\s+at|located\s+at|address\s+(?:is|was))\s+[0-9A-Za-z\s,.-]+?\b(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Way|Square|Sq|Place|Pl)\b",
    re.IGNORECASE,
)
_RESIDUE_ENDING_DIGITS_RE = re.compile(
    r"\b(?:ending\s+in|ending\s+with|acct\s*#?|account\s*#?)\s*[:\-]?\s*#?(\d{4})\b",
    re.IGNORECASE,
)
_RESIDUE_ANCHORED_INITIALS_NAME_RE = re.compile(
    r"\b(?:Dr\.|Mr\.|Ms\.|Mrs\.|Prof\.|Director|Officer|Chief|Lead|Author|Reviewer|Engineer|Architect)\s+([A-Z]\.\s*[A-Z][a-z]{2,})\b"
)
_RESIDUE_INITIALS_NAME_RE = re.compile(r"\b([A-Z]\.\s*[A-Z][a-z]{2,})\b")


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
    visual_chunks_scanned: int = 0

    def __post_init__(self):
        if not self.confirmed_leaks and self.leaks:
            self.confirmed_leaks = list(self.leaks)
        if not self.residue and self.leaks:
            self.residue = [_mask(getattr(e, "raw_value", "") or getattr(e, "masked_value", "")) for e in self.leaks]

    def summary(self) -> str:
        visual_info = f" (visual chunks: {self.visual_chunks_scanned})" if self.visual_chunks_scanned > 0 else ""
        if self.downstream_safe:
            msg = f"✅ {self.file}: downstream_safe=YES (0 leaks){visual_info}"
            if self.unverified_residue:
                msg += f" (residue items: {len(self.unverified_residue)})"
            return msg
        return (
            f"❌ {self.file}: downstream_safe=NO{visual_info} — "
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
    visual_chunks_scanned = 0

    raw_to_type, known_set = _extract_known_values(known_values)

    for chunk in redacted_chunks:
        if chunk.is_blocked():
            continue  # blocked chunks have no text — nothing to scan

        chunks_scanned += 1
        if getattr(chunk, "method", "") in ("region_ocr", "svg_text"):
            visual_chunks_scanned += 1

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

        # 3. Independent regex leak checks for residual Phase 6 entities
        # 3a. 9xx SSN
        for m in _9XX_SSN_RE.finditer(scan_text):
            if not any(s <= m.start() and m.end() <= e for (s, e) in existing_spans):
                matched_raw = m.group(0).strip()
                masked = _mask(matched_raw)
                leak_ent = Entity(
                    entity_type="SSN",
                    start=m.start(),
                    end=m.end(),
                    file=chunk.file,
                    location=chunk.location,
                    confidence=0.98,
                    sensitivity="High",
                    methods=["residual_9xx_ssn_verifier"],
                    raw_value=matched_raw,
                    masked_value=masked,
                )
                log.warning("9XX SSN LEAK in %s @ %s: %s", file_path, chunk.location, masked)
                leaks.append(leak_ent)
                existing_spans.add((m.start(), m.end()))
                if masked not in residue_set:
                    residue_set.append(masked)

        # 3b. Partial SSN pattern check (independent of known values)
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

        # 3c. International Phone
        for m in _INTL_PHONE_RE.finditer(scan_text):
            if not any(s <= m.start() and m.end() <= e for (s, e) in existing_spans):
                matched_raw = m.group(0).strip()
                digits = re.sub(r"\D", "", matched_raw)
                if len(digits) >= 7:
                    masked = _mask(matched_raw)
                    leak_ent = Entity(
                        entity_type="PHONE",
                        start=m.start(),
                        end=m.end(),
                        file=chunk.file,
                        location=chunk.location,
                        confidence=0.95,
                        sensitivity="Medium",
                        methods=["residual_intl_phone_verifier"],
                        raw_value=matched_raw,
                        masked_value=masked,
                    )
                    log.warning("INTL PHONE LEAK in %s @ %s: %s", file_path, chunk.location, masked)
                    leaks.append(leak_ent)
                    existing_spans.add((m.start(), m.end()))
                    if masked not in residue_set:
                        residue_set.append(masked)

        # 3d. Residual DOB
        for m in _RESIDUE_DOB_RE.finditer(scan_text):
            if not any(s <= m.start() and m.end() <= e for (s, e) in existing_spans):
                matched_raw = m.group(0).strip()
                masked = _mask(matched_raw)
                leak_ent = Entity(
                    entity_type="DATE_OF_BIRTH",
                    start=m.start(),
                    end=m.end(),
                    file=chunk.file,
                    location=chunk.location,
                    confidence=0.95,
                    sensitivity="High",
                    methods=["residual_dob_verifier"],
                    raw_value=matched_raw,
                    masked_value=masked,
                )
                log.warning("DOB LEAK in %s @ %s: %s", file_path, chunk.location, masked)
                leaks.append(leak_ent)
                existing_spans.add((m.start(), m.end()))
                if masked not in residue_set:
                    residue_set.append(masked)

        # 3e. Residual narrative address
        for m in _RESIDUE_NARRATIVE_ADDR_RE.finditer(scan_text):
            if not any(s <= m.start() and m.end() <= e for (s, e) in existing_spans):
                matched_raw = m.group(0).strip()
                masked = _mask(matched_raw)
                leak_ent = Entity(
                    entity_type="ADDRESS",
                    start=m.start(),
                    end=m.end(),
                    file=chunk.file,
                    location=chunk.location,
                    confidence=0.95,
                    sensitivity="Medium",
                    methods=["residual_narrative_address_verifier"],
                    raw_value=matched_raw,
                    masked_value=masked,
                )
                log.warning("NARRATIVE ADDR LEAK in %s @ %s: %s", file_path, chunk.location, masked)
                leaks.append(leak_ent)
                existing_spans.add((m.start(), m.end()))
                if masked not in residue_set:
                    residue_set.append(masked)

        # 3f. Residual ending-#### account digits
        for m in _RESIDUE_ENDING_DIGITS_RE.finditer(scan_text):
            if not any(s <= m.start() and m.end() <= e for (s, e) in existing_spans):
                matched_raw = m.group(0).strip()
                masked = _mask(matched_raw)
                leak_ent = Entity(
                    entity_type="ACCOUNT_NUMBER",
                    start=m.start(),
                    end=m.end(),
                    file=chunk.file,
                    location=chunk.location,
                    confidence=0.90,
                    sensitivity="High",
                    methods=["residual_ending_digits_verifier"],
                    raw_value=matched_raw,
                    masked_value=masked,
                )
                log.warning("ENDING DIGITS LEAK in %s @ %s: %s", file_path, chunk.location, masked)
                leaks.append(leak_ent)
                existing_spans.add((m.start(), m.end()))
                if masked not in residue_set:
                    residue_set.append(masked)

        # 3g. Anchored initials name leak
        for m in _RESIDUE_ANCHORED_INITIALS_NAME_RE.finditer(scan_text):
            if not any(s <= m.start(1) and m.end(1) <= e for (s, e) in existing_spans):
                matched_raw = m.group(1).strip()
                masked = _mask(matched_raw)
                leak_ent = Entity(
                    entity_type="PERSON",
                    start=m.start(1),
                    end=m.end(1),
                    file=chunk.file,
                    location=chunk.location,
                    confidence=0.90,
                    sensitivity="Medium",
                    methods=["residual_anchored_initials_verifier"],
                    raw_value=matched_raw,
                    masked_value=masked,
                )
                log.warning("ANCHORED INITIALS NAME LEAK in %s @ %s: %s", file_path, chunk.location, masked)
                leaks.append(leak_ent)
                existing_spans.add((m.start(1), m.end(1)))
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

        for m in _RESIDUE_INITIALS_NAME_RE.finditer(scan_text):
            if not any(s <= m.start() and m.end() <= e for (s, e) in existing_spans):
                val = m.group(0).strip()
                masked = _mask(val)
                # Avoid duplicate entry if already added
                if not any(r["masked"] == masked and r["location"] == chunk.location for r in unverified_residue):
                    unverified_residue.append({
                        "location": chunk.location,
                        "type": "INITIALS_NAME_RESIDUE",
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

        _CAP_NGRAMS_EXCLUSIONS = {
            "synthetic test data", "customer records", "not real persons", "real persons",
            "united states", "civil engineer", "credit union", "example bank", "demo credit",
            "sample savings", "audit summary", "report table", "figure token", "token recall",
            "chunk breakdown", "throughput performance", "performance speed",
        }

        for m in _RESIDUE_CAP_NGRAMS_RE.finditer(scan_text):
            if not any(s <= m.start() and m.end() <= e for (s, e) in existing_spans):
                val = m.group(0).strip()
                lower_val = val.lower()
                if lower_val in _FORM_LABEL_DENYLIST or lower_val in _CAP_NGRAMS_EXCLUSIONS:
                    continue
                words = val.split()
                if all(w.lower() in _FORM_LABEL_WORDS for w in words):
                    continue
                if any(w.lower() in (
                    "test", "dataset", "records", "data", "not", "real", "synthetic",
                    "bank", "credit", "union", "savings", "example", "demo", "sample",
                    "notes", "verified", "engineer", "nurse", "doctor", "page",
                    "revenue", "quarterly", "growth", "flow", "cache", "service", "services",
                    "system", "systems", "component", "architecture", "metrics", "chart",
                    "diagram", "table", "report", "overview", "summary", "status", "version",
                    "release", "security", "management", "operation", "operations"
                ) for w in words):
                    continue
                masked = _mask(val)
                if not any(r["masked"] == masked and r["location"] == chunk.location for r in unverified_residue):
                    unverified_residue.append({
                        "location": chunk.location,
                        "type": "CAP_NGRAMS_RESIDUE",
                        "masked": masked,
                    })
                if masked not in residue_set:
                    residue_set.append(masked)

    log.info(
        "Leakage scan complete: %d chunks scanned (%d visual), %d confirmed leaks, %d residue items in %s",
        chunks_scanned, visual_chunks_scanned, len(leaks), len(unverified_residue), file_path,
    )

    has_critical_residue = any(
        r["type"] in ("LONG_DIGITS_RESIDUE", "CAP_NGRAMS_RESIDUE")
        for r in unverified_residue
    )
    is_safe = (len(leaks) == 0) and (not has_critical_residue)

    result = LeakResult(
        file=file_path,
        downstream_safe=is_safe,
        leaks=leaks,
        confirmed_leaks=leaks,
        unverified_residue=unverified_residue,
        residue=residue_set,
        visual_chunks_scanned=visual_chunks_scanned,
    )
    log.info(result.summary())
    return result
