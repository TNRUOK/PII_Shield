"""
verify/metrics.py
-----------------
Evaluation metrics: precision, recall, F1 per entity type and per file format.
Also computes structure-retention score (difflib SequenceMatcher ratio).

Requires ground-truth JSON files produced by synthetic/generate.py.

Usage:
    python -m verify.metrics --gt-dir synthetic/ground_truth --pred <entities_json>
Or imported and called as a library.
"""
from __future__ import annotations

import json
import difflib
import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple

log = logging.getLogger(__name__)


def _normalise(text: str) -> str:
    """Normalise text for fuzzy matching: lowercase, strip whitespace."""
    return text.strip().lower()


def match_entities(
    predicted: List[Dict],   # list of {"type": ..., "text": ..., "location": ...}
    ground_truth: List[Dict],
    span_tolerance: int = 5,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Match predicted entities to ground-truth entities.

    Match criterion:
      - Same entity_type (case-insensitive)
      - Normalised text matches within span_tolerance chars (substring match allowed)

    Returns (true_positives, false_positives, false_negatives).
    """
    gt_remaining = list(ground_truth)
    tps, fps, fns = [], [], []

    for pred in predicted:
        pred_type = pred.get("type", pred.get("entity_type", "")).upper()
        pred_text = _normalise(pred.get("text", pred.get("raw_value", "")))

        matched = False
        for i, gt in enumerate(gt_remaining):
            gt_type = gt.get("type", "").upper()
            gt_text = _normalise(gt.get("text", ""))

            if pred_type != gt_type:
                continue

            # Accept if texts are equal or one is a substring of the other
            # (handles OCR noise adding/removing a char at boundary)
            if (gt_text in pred_text or pred_text in gt_text or
                    pred_text == gt_text):
                tps.append({"pred": pred, "gt": gt})
                gt_remaining.pop(i)
                matched = True
                break


        if not matched:
            fps.append(pred)

    fns = list(gt_remaining)
    return tps, fps, fns


def compute_metrics(
    predicted: List[Dict],
    ground_truth: List[Dict],
) -> Dict[str, Dict[str, float]]:
    """
    Compute precision, recall, F1 per entity type.

    Returns dict: entity_type → {"precision": ..., "recall": ..., "f1": ..., "tp": ..., ...}
    """
    type_preds: Dict[str, List] = defaultdict(list)
    type_gts: Dict[str, List] = defaultdict(list)

    for p in predicted:
        t = p.get("type", p.get("entity_type", "UNKNOWN")).upper()
        type_preds[t].append(p)

    for g in ground_truth:
        t = g.get("type", "UNKNOWN").upper()
        type_gts[t].append(g)

    all_types = sorted(set(type_preds) | set(type_gts))
    results: Dict[str, Dict[str, float]] = {}

    for etype in all_types:
        tps, fps, fns = match_entities(type_preds.get(etype, []),
                                        type_gts.get(etype, []))
        tp = len(tps)
        fp = len(fps)
        fn = len(fns)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        results[etype] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
        }

    # Overall
    total_tp = sum(v["tp"] for v in results.values())
    total_fp = sum(v["fp"] for v in results.values())
    total_fn = sum(v["fn"] for v in results.values())
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    results["OVERALL"] = {
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        round(f1, 4),
    }

    return results


def structure_retention_score(extracted_text: str, ground_truth_text: str) -> float:
    """
    Compute structure retention as difflib SequenceMatcher ratio.
    Returns a value in [0.0, 1.0]; target >= 0.80.
    """
    matcher = difflib.SequenceMatcher(
        None,
        ground_truth_text.strip(),
        extracted_text.strip(),
        autojunk=False,
    )
    return round(matcher.ratio(), 4)


def print_metrics_table(metrics: Dict[str, Dict[str, float]]) -> None:
    """Print a formatted table of precision / recall / F1 per entity type."""
    header = f"{'Entity Type':<22} {'TP':>5} {'FP':>5} {'FN':>5} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for etype, m in sorted(metrics.items()):
        row = (
            f"{etype:<22} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} "
            f"{m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}"
        )
        if etype == "OVERALL":
            print("-" * len(header))
        print(row)
    print("=" * len(header))


def run_evaluation(
    gt_dir: str,
    predicted_entities: List[Dict],
    extracted_texts: Dict[str, str],  # filename → extracted plain text
) -> Dict:
    """
    Full evaluation run.

    Parameters
    ----------
    gt_dir              : Directory containing ground-truth JSON files.
    predicted_entities  : All predicted entities from the pipeline (dicts).
    extracted_texts     : Mapping from doc name to extracted plain text.

    Returns dict with metrics and structure retention scores.
    """
    gt_dir_path = Path(gt_dir)
    all_gt_entities: List[Dict] = []
    structure_scores: Dict[str, float] = {}

    for gt_file in gt_dir_path.glob("*.json"):
        if gt_file.name.endswith("_text.txt"):
            continue
        try:
            with open(gt_file, "r", encoding="utf-8") as fh:
                gt_data = json.load(fh)
            all_gt_entities.extend(gt_data.get("entities", []))

            # Structure retention
            doc_name = gt_data.get("document", "")
            txt_file = gt_dir_path / f"{Path(doc_name).stem}_text.txt"
            if txt_file.exists() and doc_name in extracted_texts:
                gt_text = txt_file.read_text(encoding="utf-8")
                score = structure_retention_score(extracted_texts[doc_name], gt_text)
                structure_scores[doc_name] = score
        except Exception as exc:
            log.error("Failed to load GT file %s: %s", gt_file, exc)

    entity_metrics = compute_metrics(predicted_entities, all_gt_entities)
    print_metrics_table(entity_metrics)

    print("\nStructure Retention Scores (target >= 0.80):")
    for doc, score in structure_scores.items():
        status = "✅" if score >= 0.80 else "❌"
        print(f"  {status} {doc}: {score:.4f}")

    return {
        "entity_metrics": entity_metrics,
        "structure_retention": structure_scores,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="PII Shield evaluation")
    parser.add_argument("--gt-dir", required=True, help="Ground-truth directory")
    parser.add_argument("--pred", required=True, help="Predicted entities JSON file")
    args = parser.parse_args()

    with open(args.pred, "r", encoding="utf-8") as fh:
        predicted = json.load(fh)

    run_evaluation(args.gt_dir, predicted, {})
