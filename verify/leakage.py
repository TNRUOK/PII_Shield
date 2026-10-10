"""
verify/leakage.py
-----------------
Leakage check: re-run the full detection pipeline on already-redacted chunks.

Any detection hit that is NOT a placeholder token ([TYPE_N] pattern) is a LEAK.
A leak causes the file to be marked downstream_safe=False.

This is the final safety gate before content is cleared for LLM use.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List

from extract.models import Chunk
from detect.merge import detect_chunk, Entity

log = logging.getLogger(__name__)

# Matches our placeholder format [TYPE_N] — used to exclude them from re-detection
_PLACEHOLDER_RE = re.compile(r"\[[A-Z_]+_\d+\]")


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

    def summary(self) -> str:
        if self.downstream_safe:
            return f"✅ {self.file}: downstream_safe=YES (0 leaks)"
        return (
            f"❌ {self.file}: downstream_safe=NO — "
            f"{len(self.leaks)} leak(s): "
            + ", ".join(f"{e.entity_type}@{e.location}" for e in self.leaks[:5])
        )


def check_leakage(
    redacted_chunks: List[Chunk],
    config: dict,
) -> LeakResult:
    """
    Run detection on redacted chunks. Any hit outside a placeholder is a leak.

    Parameters
    ----------
    redacted_chunks : Chunks after redaction has been applied.
    config          : Loaded config dict.

    Returns
    -------
    LeakResult with downstream_safe flag and list of leaked entities.
    """
    if not redacted_chunks:
        return LeakResult(file="", downstream_safe=True, leaks=[])

    file_path = redacted_chunks[0].file
    leaks: List[Entity] = []
    chunks_scanned = 0

    for chunk in redacted_chunks:
        if chunk.is_blocked():
            continue  # blocked chunks have no text — nothing to scan

        chunks_scanned += 1
        # Strip placeholders before detection to avoid false alarm on placeholder text
        scan_text = _strip_placeholders(chunk.text)

        # Use SimpleNamespace to avoid fragile anonymous-class construction
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

        if entities:
            for ent in entities:
                log.warning(
                    "LEAK detected in %s @ %s: %s (conf=%.2f)",
                    file_path, chunk.location, ent.entity_type, ent.confidence
                )
            leaks.extend(entities)

    log.info(
        "Leakage scan complete: %d chunks scanned, %d post-redaction entities found in %s",
        chunks_scanned, len(leaks), file_path,
    )

    result = LeakResult(
        file=file_path,
        downstream_safe=len(leaks) == 0,
        leaks=leaks,
    )
    log.info(result.summary())
    return result
