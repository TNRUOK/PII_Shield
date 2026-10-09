"""
redact/mapping.py
-----------------
Manage placeholder ↔ raw-value mappings per document.

The mapping is written to a local JSON file suffixed with _mapping.json.
This file is NEVER passed to reports, the Gradio UI, or any downstream system.

Key design decisions:
  - Same raw value within one document → same placeholder token (consistent).
  - Counter is per (entity_type, document) to keep numbers readable.
  - The mapping file path is derived from the source file path so it lives
    alongside the redacted output.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

log = logging.getLogger(__name__)


class PlaceholderMapping:
    """
    Tracks raw ↔ placeholder assignments for one document.
    Thread-safe enough for single-threaded pipeline use.
    """

    def __init__(self, source_file: str, output_dir: str):
        self._raw_to_placeholder: Dict[Tuple[str, str], str] = {}
        self._placeholder_to_raw: Dict[str, str] = {}
        self._counters: Dict[str, int] = {}
        self._source_file = source_file

        # Determine mapping file path
        stem = Path(source_file).stem
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.mapping_path = out / f"{stem}_mapping.json"

    def get_or_create(self, entity_type: str, raw_value: str) -> str:
        """
        Return the placeholder for (entity_type, raw_value), creating one if new.
        Consistency: the same raw value and type always get the same placeholder.
        """
        key = (entity_type, raw_value)
        if key in self._raw_to_placeholder:
            return self._raw_to_placeholder[key]

        n = self._counters.get(entity_type, 0) + 1
        self._counters[entity_type] = n
        placeholder = f"[{entity_type}_{n}]"

        self._raw_to_placeholder[key] = placeholder
        self._placeholder_to_raw[placeholder] = raw_value
        return placeholder

    def save(self) -> None:
        """Persist mapping to the local JSON file (raw values are in this file only)."""
        data = {
            "source_file": self._source_file,
            "WARNING": "This file contains raw PII values. Do NOT share or include in reports.",
            "mappings": {
                ph: raw for ph, raw in self._placeholder_to_raw.items()
            },
        }
        with open(self.mapping_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        log.info("Mapping saved to %s (%d entries)", self.mapping_path, len(data["mappings"]))

    def load(self) -> None:
        """Load an existing mapping file (for resuming or verification)."""
        if not self.mapping_path.exists():
            return
        with open(self.mapping_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for ph, raw in data.get("mappings", {}).items():
            self._placeholder_to_raw[ph] = raw
            # Reconstruct type from placeholder "[TYPE_N]"
            etype = ph.strip("[]").rsplit("_", 1)[0]
            self._raw_to_placeholder[(etype, raw)] = ph
        log.info("Mapping loaded from %s", self.mapping_path)
