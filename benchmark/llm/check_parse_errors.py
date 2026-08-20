"""check_parse_errors.py

Report tag-parsing and verbatim-copy failures across one or more annotate.py
prediction CSVs.

Why this exists: evaluate.py scores only (label, start, end) exact-span matches, and
annotate.py's extract_entities swallows a TagParseError into zero entities (see
common.py). Neither surfaces *why* a row scored badly, so two very different failures
are invisible in the P/R/F1 table:

  1. TagParseError -- unbalanced/mismatched tags. Stored as "[]", which load_pred
     reads as an ordinary empty prediction, indistinguishable from a model that
     correctly found nothing. Its gold spans become false negatives.

  2. Text drift -- strip_tags(predicted) != raw_text. NOT checked anywhere in the
     benchmark: parse_and_verify (pipeline/parser.py), which enforces the
     character-preservation invariant, is never called on benchmark output. Offsets
     are computed against the model's own text but compared to gold offsets computed
     against raw_text, so any edit before a span shifts every following offset. Such
     a row contributes false positives AND false negatives at once -- a double
     penalty on precision and recall that reads as "bad annotation" in the report.

Drift is classified by position, because not all of it corrupts offsets:

    trailing   stripped output starts with raw_text (model appended commentary).
               Spans inside raw_text keep valid offsets -- benign.
    leading    raw_text appears in the output but not at position 0 (preamble
               prepended). Every offset is shifted by a constant.
    mutated    raw_text does not appear intact. The review text itself was altered
               (paraphrase, normalized punctuation, dropped "<br />"). Offsets after
               the first edit are unrecoverable.

A row is counted at most once per category. FAILED rows (generation gave up after
--max-retries, see annotate.py) are excluded from the parse/drift denominators and
reported separately, since they have no output to parse.

With --gold, P/R/F1 is also reported under three scoring conventions, because no
single one is honest on its own when models differ in drift rate:

    strict      What evaluate.py does today: violation rows keep their spans, which
                sit at shifted offsets, so they cost precision AND recall. The
                end-to-end number for using the model's output as-is.
    invariant   Violation rows mapped to [] -- treated exactly like FAILED. Gold
                spans become false negatives, no false positives. This is what the
                labeling pipeline would do, since parse_and_verify rejects the row
                and routes it to human review.
    clean       Only rows satisfying strip_tags(pred) == raw_text are scored, on both
                sides. Isolates labeling quality from text fidelity -- but the clean
                subset is not a random sample (longer, punctuation-heavy rows drift
                more), so read it against its row count, printed alongside.

"Violation" here means any invariant breach: a TagParseError or any drift, including
benign trailing drift, matching what parse_and_verify would reject.

Usage:
    python benchmark/llm/check_parse_errors.py predictions_*.csv
    python benchmark/llm/check_parse_errors.py a.csv b.csv --examples 3
    python benchmark/llm/check_parse_errors.py a.csv --labels MINOR_AGE MINOR_EDU FAM_KIN
    python benchmark/llm/check_parse_errors.py predictions_*.csv \
        --gold output/test_balanced_400.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# evaluate.py is a sibling; pipeline/parser.py lives at the repo root (three parents up).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from evaluate import (COARSE_3DIM, FAILED_MARKER, THREE_DIM_LABELS,  # noqa: E402
                      load_gold, load_pred, score)
from pipeline.parser import TagParseError, parse_tagged_text, strip_tags  # noqa: E402

REQUIRED_COLS = ["row_id", "raw_text", "predicted_tagged_text", "predicted_entities_json"]


def first_divergence(a: str, b: str) -> int:
    """Index of the first differing character, or len of the shorter if one is a prefix."""
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


def classify_drift(stripped: str, raw: str) -> str:
    """'trailing' | 'leading' | 'mutated' -- see module docstring."""
    if stripped.startswith(raw):
        return "trailing"
    if raw in stripped:
        return "leading"
    return "mutated"


def analyze(path: Path, labels: list[str]) -> dict:
    """Per-file counts. Raises ValueError with a readable message on a bad CSV."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing column(s): {', '.join(missing)}")

    df["predicted_tagged_text"] = df["predicted_tagged_text"].fillna("")
    df["raw_text"] = df["raw_text"].fillna("")
    is_failed = df["predicted_entities_json"].astype(str) == FAILED_MARKER

    stats = {
        "path": path,
        "rows": len(df),
        "failed": int(is_failed.sum()),
        "parse_error": 0,
        "trailing": 0,
        "leading": 0,
        "mutated": 0,
        "drift_with_tags": 0,
        "clean": 0,
        "examples": [],
        # Row-id sets for the --gold scoring variants. "violation" = any invariant
        # breach (parse error or drift); "clean" = passes strip_tags(pred) == raw.
        "violation_ids": set(),
        "clean_ids": set(),
        "failed_ids": {str(r) for r in df.loc[is_failed, "row_id"]},
    }

    for _, row in df[~is_failed].iterrows():
        tagged, raw = row["predicted_tagged_text"], row["raw_text"]
        rid = str(row["row_id"])

        try:
            parse_tagged_text(tagged, tagset=labels)
        except TagParseError as exc:
            stats["parse_error"] += 1
            stats["violation_ids"].add(rid)
            stats["examples"].append((rid, "parse_error", str(exc)))
            # Tags are unbalanced, so a drift check on this row would just re-report
            # the same defect; count it once, under the more specific category.
            continue

        stripped = strip_tags(tagged, tagset=labels)
        if stripped == raw:
            stats["clean"] += 1
            stats["clean_ids"].add(rid)
            continue

        kind = classify_drift(stripped, raw)
        stats[kind] += 1
        stats["violation_ids"].add(rid)
        if row.get("num_predicted_entities", 0) > 0:
            stats["drift_with_tags"] += 1
        at = first_divergence(stripped, raw)
        stats["examples"].append((
            rid, kind,
            f"diverges at char {at}: gold {raw[at:at + 40]!r} vs pred {stripped[at:at + 40]!r}",
        ))

    stats["scored"] = stats["rows"] - stats["failed"]
    stats["drift"] = stats["trailing"] + stats["leading"] + stats["mutated"]
    stats["offset_corrupting"] = stats["leading"] + stats["mutated"]
    return stats


def pct(n: int, total: int) -> str:
    return f"{n:>4} ({n / total:>5.1%})" if total else f"{n:>4}     -"


def score_variants(gold: dict, stats: dict, collapse: bool) -> dict:
    """P/R/F1 under the strict / invariant / clean conventions -- see module docstring."""
    pred, failed = load_pred(str(stats["path"]))
    label_map = COARSE_3DIM if collapse else None
    violations = stats["violation_ids"]

    # strict: exactly what evaluate.py scores today, violations included as-is.
    strict = score(gold, pred, failed, gold_label_map=label_map)

    # invariant: violation rows lose their (mis-offset) spans, like FAILED rows.
    inv_pred = {rid: ([] if rid in violations else ents) for rid, ents in pred.items()}
    invariant = score(gold, inv_pred, failed | violations, gold_label_map=label_map)

    # clean: restrict BOTH sides to rows that satisfied the invariant.
    keep = stats["clean_ids"]
    clean = score({k: v for k, v in gold.items() if k in keep},
                  {k: v for k, v in pred.items() if k in keep},
                  set(), gold_label_map=label_map)

    return {"strict": strict, "invariant": invariant, "clean": clean}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Report tag-parse errors and verbatim-copy drift across prediction CSVs.")
    ap.add_argument("csvs", nargs="+", help="One or more annotate.py --out-csv files.")
    ap.add_argument("--labels", nargs="+", default=THREE_DIM_LABELS,
                    help=f"Tagset to parse against (default: {' '.join(THREE_DIM_LABELS)}). "
                         "Must match the label scheme the guideline told the model to emit.")
    ap.add_argument("--examples", type=int, default=0, metavar="N",
                    help="Also print the first N offending rows per file, with the character "
                         "offset where the prediction diverges from raw_text.")
    ap.add_argument("--gold", default=None,
                    help="Gold CSV (row_id + entities_json), e.g. the --input-csv the run was "
                         "annotated from. Adds P/R/F1 under the strict / invariant / clean "
                         "conventions described in the module docstring.")
    ap.add_argument("--no-collapse", action="store_true",
                    help="Do not fold gold's 5 fine labels onto MINOR/GENDER/KINSHIP before "
                         "scoring. Pass this only if --gold already uses the same label scheme "
                         "as the predictions.")
    args = ap.parse_args()

    results, errors = [], []
    for name in args.csvs:
        path = Path(name)
        try:
            results.append(analyze(path, args.labels))
        except FileNotFoundError:
            errors.append((path, "file not found"))
        except ValueError as exc:
            errors.append((path, str(exc)))
        except pd.errors.EmptyDataError:
            errors.append((path, "empty file"))

    if results:
        width = max(len(r["path"].name) for r in results)
        header = (f"{'file':<{width}} {'rows':>5} {'FAILED':>13} {'parse_err':>13} "
                  f"{'drift':>13} {'offset_bad':>13} {'clean':>13}")
        print(f"\nParsed against tagset: {' '.join(args.labels)}")
        print(header)
        print("-" * len(header))
        for r in results:
            n = r["scored"]
            print(f"{r['path'].name:<{width}} {r['rows']:>5} "
                  f"{pct(r['failed'], r['rows'])} {pct(r['parse_error'], n)} "
                  f"{pct(r['drift'], n)} {pct(r['offset_corrupting'], n)} "
                  f"{pct(r['clean'], n)}")

        print("\nFAILED is a % of all rows; the rest are % of non-FAILED (parseable) rows.")
        print("drift       = strip_tags(prediction) != raw_text (any position)")
        print("offset_bad  = drift that shifts span offsets (leading + mutated); "
              "trailing drift is benign")

        print(f"\n{'file':<{width}} {'trailing':>9} {'leading':>9} {'mutated':>9} "
              f"{'drift_w_tags':>13}")
        print("-" * (width + 43))
        for r in results:
            print(f"{r['path'].name:<{width}} {r['trailing']:>9} {r['leading']:>9} "
                  f"{r['mutated']:>9} {r['drift_with_tags']:>13}")
        print("\ndrift_w_tags = drifted rows that still emitted tags, i.e. rows actively "
              "contributing\n               spans at wrong offsets (false positives AND "
              "false negatives).")

    if args.gold and results:
        gold = load_gold(args.gold)
        collapse = not args.no_collapse
        print(f"\nScored against {args.gold}"
              + (" (gold's 5 labels folded to MINOR/GENDER/KINSHIP)" if collapse else ""))
        width = max(len(r["path"].name) for r in results)
        header = (f"{'file':<{width}} {'convention':<10} {'rows':>5} {'TP':>5} {'FP':>5} "
                  f"{'FN':>5} {'P':>7} {'R':>7} {'F1':>7}")
        print(header)
        print("-" * len(header))
        for r in results:
            variants = score_variants(gold, r, collapse)
            for name, res in variants.items():
                m = res["micro"]
                print(f"{r['path'].name if name == 'strict' else '':<{width}} {name:<10} "
                      f"{res['rows_scored']:>5} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} "
                      f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f}")
            print()
        print("strict    = violations scored as-is (what evaluate.py reports today)")
        print("invariant = violations mapped to [], like FAILED (what the pipeline would do)")
        print("clean     = only invariant-passing rows scored; compare 'rows' across files, "
              "since\n            the clean subset is not a random sample")

    if args.examples:
        for r in results:
            if not r["examples"]:
                continue
            print(f"\n--- {r['path'].name}: first {min(args.examples, len(r['examples']))} "
                  f"of {len(r['examples'])} offending row(s) ---")
            for row_id, kind, detail in r["examples"][:args.examples]:
                print(f"  [{kind}] {row_id}\n      {detail}")

    if errors:
        print("\nSkipped:")
        for path, why in errors:
            print(f"  {path}: {why}")
        if not results:
            sys.exit(1)


if __name__ == "__main__":
    main()
