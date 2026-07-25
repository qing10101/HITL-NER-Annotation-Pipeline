"""
evaluate.py

Scores annotate.py prediction CSV(s) against a gold-standard CSV, computing
entity-level precision/recall/F1 per label (exact span match: label + start +
end offset, same convention as entities_json). Supports scoring several named
prediction sets against the same gold set side by side — this is how the
retriever-equipped (--k 8) and zero-shot (--k 0) conditions get compared:
run annotate.py once per condition, then score both here.

Rows annotate.py could not produce output for (all retries exhausted) are marked
FAILED in its output CSV. They are scored as recall misses — every gold entity on
a failed row becomes a false negative, and no false positives are attributed — and
their count is reported explicitly in each per-condition report and the comparison
table, so a run's failures are never silently hidden inside the F1.

Usage:
    # Score one prediction set
    python evaluate.py --gold validation/general/gold_standard_merged.csv \
        --pred predictions.csv

    # Compare zero-shot vs retriever-equipped conditions against the same gold set
    python evaluate.py --gold validation/general/gold_standard_merged.csv \
        --pred zero_shot=predictions_k0.csv \
        --pred retriever=predictions_k8.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

# Sentinel written to predicted_entities_json when annotate.py exhausts all retries
# for a row. Distinguishes a hard failure (no output produced) from a genuine
# empty prediction ("[]", model correctly found no entities). Not valid JSON on
# purpose, so it can never collide with a real prediction.
FAILED_MARKER = "FAILED"


def load_gold(path: str) -> dict[str, list[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        return {
            row["row_id"]: json.loads(row["entities_json"] or "[]")
            for row in csv.DictReader(f)
        }


def load_pred(path: str) -> tuple[dict[str, list[dict]], set[str]]:
    """Load predictions -> (entities_by_id, failed_ids).

    A row marked FAILED (all retries exhausted in annotate.py) maps to [] in
    entities_by_id — so its gold spans score as false negatives (a recall miss) —
    and its id is also collected in failed_ids for explicit reporting.
    """
    pred: dict[str, list[dict]] = {}
    failed: set[str] = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = row["row_id"]
            raw = row["predicted_entities_json"]
            if raw == FAILED_MARKER:
                pred[rid] = []
                failed.add(rid)
            else:
                pred[rid] = json.loads(raw or "[]")
    return pred, failed


def _span_key(ent: dict) -> tuple[str, int, int]:
    return (ent["label"], ent["start"], ent["end"])


def score(gold_by_id: dict[str, list[dict]], pred_by_id: dict[str, list[dict]],
          failed_ids: set[str] | None = None) -> dict:
    """Entity-level exact-span (label, start, end) precision/recall/F1, per label + micro.

    ``failed_ids`` are rows annotate.py could not produce output for (all retries
    exhausted). They must already be present in ``pred_by_id`` as [] so their gold
    spans count as false negatives; this argument is only for explicit reporting of
    how many scored rows were failures (and the gold entities lost with them).
    """
    failed_ids = failed_ids or set()
    shared_ids = sorted(set(gold_by_id) & set(pred_by_id))
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()

    for row_id in shared_ids:
        gold_spans = {_span_key(e) for e in gold_by_id[row_id]}
        pred_spans = {_span_key(e) for e in pred_by_id[row_id]}
        for key in pred_spans & gold_spans:
            tp[key[0]] += 1
        for key in pred_spans - gold_spans:
            fp[key[0]] += 1
        for key in gold_spans - pred_spans:
            fn[key[0]] += 1

    labels = sorted(set(tp) | set(fp) | set(fn))
    per_label = {}
    for lbl in labels:
        p = tp[lbl] / (tp[lbl] + fp[lbl]) if (tp[lbl] + fp[lbl]) else 0.0
        r = tp[lbl] / (tp[lbl] + fn[lbl]) if (tp[lbl] + fn[lbl]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_label[lbl] = {"tp": tp[lbl], "fp": fp[lbl], "fn": fn[lbl], "precision": p, "recall": r, "f1": f1}

    total_tp, total_fp, total_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    failed_in_scope = failed_ids & set(gold_by_id)
    failed_gold_entities = sum(len(gold_by_id[r]) for r in failed_in_scope)

    return {
        "rows_scored": len(shared_ids),
        "rows_missing_from_pred": len(set(gold_by_id) - set(pred_by_id)),
        "rows_missing_from_gold": len(set(pred_by_id) - set(gold_by_id)),
        "failed_rows": len(failed_in_scope),
        "failed_gold_entities": failed_gold_entities,
        "per_label": per_label,
        "micro": {"tp": total_tp, "fp": total_fp, "fn": total_fn,
                  "precision": micro_p, "recall": micro_r, "f1": micro_f1},
    }


def print_report(name: str, result: dict) -> None:
    print(f"\n=== {name} ===")
    note = []
    if result["rows_missing_from_pred"]:
        note.append(f"{result['rows_missing_from_pred']} gold rows missing from predictions")
    if result["rows_missing_from_gold"]:
        note.append(f"{result['rows_missing_from_gold']} predicted rows missing from gold")
    suffix = f"  ({'; '.join(note)})" if note else ""
    print(f"Rows scored: {result['rows_scored']}{suffix}")

    failed = result.get("failed_rows", 0)
    if failed:
        print(f"  of which FAILED (all retries exhausted): {failed} row(s), "
              f"{result['failed_gold_entities']} gold entit"
              f"{'y' if result['failed_gold_entities'] == 1 else 'ies'} "
              f"counted as false negatives")

    print(f"{'Label':<14} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>7} {'R':>7} {'F1':>7}")
    print("-" * 55)
    for lbl, m in sorted(result["per_label"].items()):
        print(f"{lbl:<14} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} "
              f"{m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f}")
    m = result["micro"]
    print("-" * 55)
    print(f"{'MICRO-AVG':<14} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} "
          f"{m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f}")


def parse_pred_arg(raw: str) -> tuple[str, str]:
    """'foo.csv' -> name 'foo'; 'name=foo.csv' -> name 'name'."""
    if "=" in raw:
        name, path = raw.split("=", 1)
        return name, path
    return Path(raw).stem, raw


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score annotate.py predictions against gold; compare multiple "
                     "conditions (e.g. zero-shot --k 0 vs. retriever-equipped --k 8)."
    )
    ap.add_argument("--gold", required=True, help="Gold-standard CSV (row_id, entities_json, ...).")
    ap.add_argument(
        "--pred", action="append", required=True,
        help="Predictions CSV to score, optionally 'name=path.csv' to label it in the "
             "report (default label: the filename stem). Repeatable to compare conditions.",
    )
    args = ap.parse_args()

    gold = load_gold(args.gold)

    results = {}
    for raw in args.pred:
        name, path = parse_pred_arg(raw)
        pred, failed = load_pred(path)
        results[name] = score(gold, pred, failed)
        print_report(name, results[name])

    if len(results) > 1:
        print("\n=== COMPARISON (micro-avg) ===")
        width = max(len(name) for name in results)
        for name, result in results.items():
            m = result["micro"]
            failed = result.get("failed_rows", 0)
            fail_note = f"  [{failed} failed row(s)]" if failed else ""
            print(f"  {name:<{width}}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
                  f"F1={m['f1']:.3f}{fail_note}")


if __name__ == "__main__":
    main()
