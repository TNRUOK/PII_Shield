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


def _image_coverage(chunks: List[Chunk]) -> Dict[str, Any]:
    """Compute image/region coverage metrics from chunk list."""
    region_chunks = [c for c in chunks if c.method in ("region_ocr", "svg_text") or c.image_status is not None]
    total_img = len(region_chunks)
    read_count = sum(1 for c in region_chunks if c.image_status == "READ")
    low_conf = sum(1 for c in region_chunks if c.image_status == "LOW_CONFIDENCE")
    unread = sum(1 for c in region_chunks if c.image_status in ("UNREAD", "UNVERIFIED", "VECTOR_UNSUPPORTED"))

    # Compute unverified region area in pixels if bbox available
    unverified_area = sum(
        (c.region_bbox[2] * c.region_bbox[3])
        for c in region_chunks
        if c.image_status not in ("READ",) and c.region_bbox
    )
    total_area = sum(
        (c.region_bbox[2] * c.region_bbox[3])
        for c in region_chunks
        if c.region_bbox
    )
    area_fraction_unverified = unverified_area / max(total_area, 1) if total_img > 0 else 0.0

    return {
        "total_image_regions": total_img,
        "read": read_count,
        "low_confidence": low_conf,
        "unread": unread,
        "unverified_area_fraction": round(area_fraction_unverified, 4),
    }


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
    region_ocr_count = sum(1 for c in chunks if c.method == "region_ocr")
    svg_count = sum(1 for c in chunks if c.method == "svg_text")
    blocked_count = sum(1 for c in chunks if c.is_blocked())

    # PII count by type
    pii_by_type: Dict[str, int] = {}
    for ent in entities:
        pii_by_type[ent.entity_type] = pii_by_type.get(ent.entity_type, 0) + 1

    coverage = _image_coverage(chunks)

    # multimodal_safe = False if any image region is unverified/unread/vector_unsupported
    multimodal_safe = (coverage["unread"] == 0)

    return {
        "file": file_path,
        "chunk_count": total,
        "native_chunks": native_count,
        "ocr_chunks": ocr_count,
        "region_ocr_chunks": region_ocr_count,
        "svg_chunks": svg_count,
        "blocked_chunks": blocked_count,
        "pii_count": len(entities),
        "suspect_ocr_count": pii_by_type.get("SUSPECT_OCR", 0),
        "pii_by_type": pii_by_type,
        "risk_level": _file_risk(entities),
        "structure_retention_score": structure_score,
        "image_coverage": coverage,
        "multimodal_safe": multimodal_safe,
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


def _startup_check_log() -> str:
    """Log and return startup check for available OCR engines and converters."""
    from extract.docx import is_vector_converter_available
    has_conv, conv_tool = is_vector_converter_available()
    tess = "✅ Tesseract" if is_tesseract_available() else "⚠️ Tesseract NOT found"
    conv = f"✅ {conv_tool}" if has_conv else "⚠️ EMF/WMF converter NOT found (no soffice/inkscape)"
    msg = f"Startup: OCR={tess} | VectorConverter={conv}"
    log.info(msg)
    return msg


def print_summary_table(file_summaries: List[Dict[str, Any]]) -> str:
    """Return a human-readable summary table as a string."""
    lines = []
    lines.append("\n" + "=" * 105)
    lines.append(_startup_check_log())
    lines.append("-" * 105)
    lines.append(
        f"{'File':<28} {'Chunks':>6} {'Blk':>4} {'PII':>5} {'Susp':>5} "
        f"{'Risk':<7} {'Struct':>7} {'Safe?':<6} {'MM-Safe':>7} {'Img':>5}"
    )
    lines.append("-" * 105)
    for fs in file_summaries:
        name = Path(fs["file"]).name[:26]
        struct = f"{fs['structure_retention_score']:.2f}" if fs.get("structure_retention_score") is not None else "  N/A"
        safe = "✅ YES" if fs["leak_check"]["downstream_safe"] else "❌ NO "
        susp = fs.get("suspect_ocr_count", 0)
        mm_safe = "✅ YES" if fs.get("multimodal_safe", True) else "⚠️ NO "
        img_total = fs.get("image_coverage", {}).get("total_image_regions", 0)
        lines.append(
            f"{name:<28} {fs['chunk_count']:>6} {fs['blocked_chunks']:>4} "
            f"{fs['pii_count']:>5} {susp:>5} {fs['risk_level']:<7} {struct:>7} "
            f"{safe:<6} {mm_safe:>7} {img_total:>5}"
        )
    lines.append("=" * 105)
    table = "\n".join(lines)
    print(table)
    return table
