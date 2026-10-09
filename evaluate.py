"""
evaluate.py
-----------
End-to-end evaluation runner:
Processes all synthetic documents, measures detection performance against
ground truth, verifies zero leakage, and prints evaluation metrics.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from extract.router import route
from detect.merge import detect_chunk, Entity
from redact.mapping import PlaceholderMapping
from redact.redactor import redact_chunks
from verify.leakage import check_leakage
from verify.metrics import run_evaluation
from report.report import build_file_summary, print_summary_table

log = logging.getLogger("evaluate")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROOT = Path(__file__).resolve().parent
SYNTHETIC_DIR = ROOT / "synthetic"
GT_DIR = SYNTHETIC_DIR / "ground_truth"


def main():
    print("=" * 70)
    print("PII SHIELD — COMPREHENSIVE PIPELINE EVALUATION")
    print("=" * 70)

    with open(ROOT / "config.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    doc_files = [
        SYNTHETIC_DIR / "training_guide.docx",
        SYNTHETIC_DIR / "org_pack.pptx",
        SYNTHETIC_DIR / "scanned_policy.pdf",
    ]

    all_predicted_entities: List[Dict] = []
    extracted_texts: Dict[str, str] = {}
    file_summaries = []

    for doc_path in doc_files:
        if not doc_path.exists():
            print(f"Skipping {doc_path.name} (not found)")
            continue

        print(f"\nProcessing: {doc_path.name}...")
        chunks = route(str(doc_path), config)

        # Reconstruct extracted text
        plain_text = "\n".join(c.text for c in chunks if not c.is_blocked())
        extracted_texts[doc_path.name] = plain_text

        # Detect PII
        entities: List[Entity] = []
        for chunk in chunks:
            if not chunk.is_blocked():
                ents = detect_chunk(chunk, config)
                entities.extend(ents)

        # Redact
        out_dir = ROOT / "output_eval"
        mapping = PlaceholderMapping(source_file=str(doc_path), output_dir=str(out_dir))
        redacted = redact_chunks(chunks, entities, mapping, config)

        # Leakage check
        leak_res = check_leakage(redacted, config)

        # File summary
        summary = build_file_summary(
            str(doc_path),
            chunks,
            entities,
            leak_result=leak_res,
            structure_score=None,
        )
        file_summaries.append(summary)

        # Collect predicted entities for metrics (using raw_value for matching vs GT)
        for e in entities:
            all_predicted_entities.append({
                "type": e.entity_type,
                "text": e.raw_value,
                "location": e.location,
                "document": doc_path.name,
            })

        print(f"  Chunks extracted : {len(chunks)} (Blocked: {summary['blocked_chunks']})")
        print(f"  PII detected     : {len(entities)}")
        print(f"  Leakage status   : {'CLEAN (Safe)' if leak_res.downstream_safe else 'LEAK DETECTED'}")

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY REPORT TABLE")
    print("=" * 70)
    print_summary_table(file_summaries)

    print("\n" + "=" * 70)
    print("BENCHMARK METRICS (VS GROUND TRUTH)")
    print("=" * 70)
    eval_results = run_evaluation(str(GT_DIR), all_predicted_entities, extracted_texts)

    eval_out = ROOT / "evaluation_results.json"
    with open(eval_out, "w", encoding="utf-8") as fh:
        json.dump(eval_results, fh, indent=2)
    print(f"\nSaved evaluation metrics to {eval_out}")


if __name__ == "__main__":
    main()
