"""
report/report.py
----------------
Generate the audit report for one or more processed files.

Outputs:
  - <timestamp>_report.csv   : Per-entity rows (masked values only, NO raw PII)
  - <timestamp>_report.json  : Same data in JSON
  - Summary printed to stdout / returned as a dict

Per-file summary includes:
  chunk count, native/OCR/blocked breakdown, PII count by type,
  risk level, structure-retention score, leak-check result, downstream_safe flag.

Per-entity row: masked_value, type, sensitivity, confidence, methods, file, location.
Raw values are NEVER written here.
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from extract.models import Chunk
from extract.ocr import is_tesseract_available
from detect.merge import Entity
from verify.leakage import LeakResult

log = logging.getLogger(__name__)

_RISK_ORDER = {"High": 3, "Medium": 2, "Low": 1}


def _file_risk(entities: List[Entity]) -> str:
    if not entities:
        return "Low"
    top = max(entities, key=lambda e: _RISK_ORDER.get(e.sensitivity, 0))
    return top.sensitivity


def build_file_summary(
    file_path: str,
    chunks: List[Chunk],
    entities: List[Entity],
    leak_result: LeakResult,
    structure_score: Optional[float],
) -> Dict[str, Any]:
    """Build a per-file summary dict (safe to serialise — no raw PII)."""
    total = len(chunks)
    native_count = sum(1 for c in chunks if c.method == "native")
    ocr_count    = sum(1 for c in chunks if c.method == "ocr")
    blocked_count = sum(1 for c in chunks if c.is_blocked())

    # PII count by type
    pii_by_type: Dict[str, int] = {}
    for ent in entities:
        pii_by_type[ent.entity_type] = pii_by_type.get(ent.entity_type, 0) + 1

    return {
        "file": file_path,
        "chunk_count": total,
        "native_chunks": native_count,
        "ocr_chunks": ocr_count,
        "blocked_chunks": blocked_count,
        "pii_count": len(entities),
        "pii_by_type": pii_by_type,
        "risk_level": _file_risk(entities),
        "structure_retention_score": structure_score,
        "leak_check": {
            "downstream_safe": leak_result.downstream_safe,
            "leak_count": len(leak_result.leaks),
        },
    }


def _entity_to_row(ent: Entity) -> Dict[str, Any]:
    """Convert an Entity to a report row. Raw value is NEVER included."""
    return {
        "file":          ent.file,
        "location":      ent.location,
        "entity_type":   ent.entity_type,
        "sensitivity":   ent.sensitivity,
        "confidence":    ent.confidence,
        "methods":       "|".join(ent.methods),
        "masked_value":  ent.masked_value,
        "placeholder":   ent.placeholder,
        "start":         ent.start,
        "end":           ent.end,
    }


def generate_report(
    file_summaries: List[Dict[str, Any]],
    all_entities: List[Entity],
    output_dir: str,
) -> Dict[str, str]:
    """
    Write CSV and JSON report files and return their paths.

    Parameters
    ----------
    file_summaries : List of dicts from build_file_summary().
    all_entities   : All Entity objects across all files.
    output_dir     : Directory to write report files into.

    Returns dict {"csv": path, "json": path}.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path  = out / f"{ts}_report.csv"
    json_path = out / f"{ts}_report.json"

    # --- CSV ---
    fieldnames = [
        "file", "location", "entity_type", "sensitivity",
        "confidence", "methods", "masked_value", "placeholder",
        "start", "end",
    ]
    entity_rows = [_entity_to_row(e) for e in all_entities]

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entity_rows)

    # --- JSON ---
    report_data = {
        "generated_at": ts,
        "tesseract_available": is_tesseract_available(),
        "WARNING": "This report contains masked PII values only. Raw values are in separate _mapping.json files.",
        "file_summaries": file_summaries,
        "entities": entity_rows,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report_data, fh, indent=2, ensure_ascii=False)

    log.info("Report written: %s, %s", csv_path, json_path)
    return {"csv": str(csv_path), "json": str(json_path)}


def print_summary_table(file_summaries: List[Dict[str, Any]]) -> str:
    """Return a human-readable summary table as a string."""
    lines = []
    lines.append("\n" + "=" * 90)
    lines.append(f"OCR Engine (Tesseract): {'Available' if is_tesseract_available() else 'Not Installed'}")
    lines.append("-" * 90)
    lines.append(f"{'File':<30} {'Chunks':>7} {'Blk':>4} {'PII':>5} {'Risk':<8} {'Struct':>7} {'Safe?':<6}")
    lines.append("-" * 90)
    for fs in file_summaries:
        name = Path(fs["file"]).name[:28]
        struct = f"{fs['structure_retention_score']:.2f}" if fs.get("structure_retention_score") is not None else "  N/A"
        safe = "✅ YES" if fs["leak_check"]["downstream_safe"] else "❌ NO "
        lines.append(
            f"{name:<30} {fs['chunk_count']:>7} {fs['blocked_chunks']:>4} "
            f"{fs['pii_count']:>5} {fs['risk_level']:<8} {struct:>7} {safe:<6}"
        )
    lines.append("=" * 90)
    table = "\n".join(lines)
    print(table)
    return table
