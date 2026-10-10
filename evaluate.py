"""
evaluate.py
-----------
End-to-end evaluation runner:
Processes all synthetic documents, measures detection performance against
ground truth, verifies zero leakage, and prints evaluation metrics.
"""
from __future__ import annotations

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="PII Shield Evaluation")
    parser.add_argument("--input", default="synthetic", help="Input directory containing documents and ground_truth/")
    parser.add_argument("--baseline", default=None, help="Path to baseline json file for comparison")
    parser.add_argument("--output", default="evaluation_results.json", help="Path to save evaluation results json")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    gt_dir = input_dir / "ground_truth"

    print("=" * 70)
    print(f"PII SHIELD — COMPREHENSIVE PIPELINE EVALUATION ({input_dir.name})")
    print("=" * 70)

    with open(ROOT / "config.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    doc_files = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in (".docx", ".pptx", ".pdf")]
    )

    all_predicted_entities: List[Dict] = []
    extracted_texts: Dict[str, str] = {}
    file_summaries = []

    for doc_path in doc_files:
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

        # Leakage check with known-value propagation
        leak_res = check_leakage(redacted, config, known_values=mapping)

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
    eval_results = run_evaluation(str(gt_dir), all_predicted_entities, extracted_texts)

    eval_out = Path(args.output)
    if not eval_out.is_absolute():
        eval_out = ROOT / eval_out
    with open(eval_out, "w", encoding="utf-8") as fh:
        json.dump(eval_results, fh, indent=2)
    print(f"\nSaved evaluation metrics to {eval_out}")

    # Baseline comparison if provided
    if args.baseline:
        b_path = Path(args.baseline)
        if not b_path.is_absolute():
            b_path = ROOT / b_path
        if b_path.exists():
            with open(b_path, "r", encoding="utf-8") as fh:
                baseline_data = json.load(fh)
            print("\n" + "=" * 70)
            print("COMPARISON VS BASELINE")
            print("=" * 70)
            b_overall = baseline_data.get("entity_metrics", {}).get("OVERALL", {})
            c_overall = eval_results.get("entity_metrics", {}).get("OVERALL", {})
            b_rec = b_overall.get("recall", 0.0)
            c_rec = c_overall.get("recall", 0.0)
            b_prec = b_overall.get("precision", 0.0)
            c_prec = c_overall.get("precision", 0.0)
            b_f1 = b_overall.get("f1", 0.0)
            c_f1 = c_overall.get("f1", 0.0)
            print(f"Overall Recall   : {b_rec:.4f} → {c_rec:.4f} ({c_rec - b_rec:+.4f})")
            print(f"Overall Precision: {b_prec:.4f} → {c_prec:.4f} ({c_prec - b_prec:+.4f})")
            print(f"Overall F1       : {b_f1:.4f} → {c_f1:.4f} ({c_f1 - b_f1:+.4f})")


if __name__ == "__main__":
    main()
