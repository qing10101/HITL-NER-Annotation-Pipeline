"""Pipeline ablations C1–C3 on an existing run, scored against human annotation.

Re-uses the artifacts of a completed run (no API calls):
  annotator_cache.csv   raw Gemini output ỹ per row
  run_log.csv           auditor verdict per row
  human_reviewed.csv    human adjudication of every row the pipeline escalated
  <human gold csv>      independent human annotation (ground truth)

Conditions (escalated rows take the human_reviewed annotation):
  C0  full pipeline        escalate if auditor FAIL or parser check fails
  C1  raw Gemini           no escalation; offsets computed on ỹ, no strip(ỹ)==x check
  C2  parser-only          escalate iff parse_and_verify(ỹ, x) fails
  C3  auditor-only         escalate iff auditor FAIL; offsets as in C1

Rows where the annotator returned nothing (PIPELINE_ERROR) have empty ỹ: they
score as no predictions in C1, fail the parser check in C2, and are escalated
in C3 because the pipeline routes hard errors to review before any audit runs.

Spans match exactly on (label, start, end); micro P/R/F1 over all rows.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.parser import TagParseError, parse_and_verify, parse_tagged_text  # noqa: E402

RUN_DIR = ROOT / "validation" / "general"
LABELS = ["MINOR_AGE", "MINOR_EDU", "GEN_NOUN", "GEN_PHYS", "FAM_KIN"]


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _spans(tagged: str) -> set[tuple[str, int, int]]:
    """Offsets computed directly on the tagged text, no invariant check (C1 parsing)."""
    if not tagged:
        return set()
    try:
        _, spans = parse_tagged_text(tagged)
    except TagParseError:
        return set()
    return {(s.label, s.start, s.end) for s in spans}


def _parser_rejects(tagged: str, raw: str) -> bool:
    try:
        parse_and_verify(tagged, raw)
    except TagParseError:
        return True
    return False


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, default=RUN_DIR)
    p.add_argument("--human-csv", type=Path, default=RUN_DIR / "sample_500_human.csv",
                   help="Ground-truth CSV with row_id, raw text and tagged text columns")
    p.add_argument("--raw-col", default="text", help="Raw text column in --human-csv")
    p.add_argument("--gold-col", default="annotated_text", help="Tagged text column in --human-csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    truth = {r["row_id"]: r for r in _read(args.human_csv)}
    gemini = {r["row_id"]: r["tagged_text"] for r in _read(args.run_dir / "annotator_cache.csv")}
    log = {r["row_id"]: r for r in _read(args.run_dir / "run_log.csv")}
    reviewed = {r["row_id"]: r["human_annotation"] for r in _read(args.run_dir / "human_reviewed.csv")}

    ids = sorted(truth)
    gold = {rid: _spans(truth[rid][args.gold_col]) for rid in ids}
    raw = {rid: truth[rid][args.raw_col] for rid in ids}
    y = {rid: gemini.get(rid, "") for rid in ids}

    auditor_fail = {rid for rid in ids if log[rid]["status"] == "FAIL"}
    parser_fail = {rid for rid in ids if _parser_rejects(y[rid], raw[rid])}

    conditions = {
        "C0 full pipeline": auditor_fail | parser_fail,
        "C1 raw Gemini": set(),
        "C2 parser-only": parser_fail,
        "C3 auditor-only": auditor_fail,
    }

    missing = (auditor_fail | parser_fail) - set(reviewed)
    if missing:
        sys.exit(f"Escalated rows without a human_reviewed annotation: {sorted(missing)}")

    print(f"Rows: {len(ids)}   ground-truth spans: {sum(map(len, gold.values()))}")
    print(f"Parser rejects raw ỹ: {sorted(parser_fail)}")
    print(f"Auditor FAIL rows   : {len(auditor_fail)}\n")

    header = f"{'Condition':<18} {'Esc':>4} {'Rate':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>7} {'R':>7} {'F1':>7}"
    print(header)
    print("-" * len(header))
    per_label: dict[str, dict[str, Counter]] = {}
    for name, escalated in conditions.items():
        tp = fp = fn = 0
        lab: dict[str, Counter] = {k: Counter() for k in ("tp", "fp", "fn")}
        for rid in ids:
            pred = _spans(reviewed[rid]) if rid in escalated else _spans(y[rid])
            g = gold[rid]
            for kind, s in (("tp", pred & g), ("fp", pred - g), ("fn", g - pred)):
                for lbl, _, _ in s:
                    lab[kind][lbl] += 1
            tp += len(pred & g)
            fp += len(pred - g)
            fn += len(g - pred)
        per_label[name] = lab
        p, r, f = _prf(tp, fp, fn)
        print(f"{name:<18} {len(escalated):>4} {len(escalated)/len(ids):>6.1%} "
              f"{tp:>4} {fp:>4} {fn:>4} {p:>7.4f} {r:>7.4f} {f:>7.4f}")

    print("\nPer-label F1")
    print(f"{'Label':<11}" + "".join(f"{n.split()[0]:>8}" for n in conditions))
    for lbl in LABELS:
        cells = []
        for name in conditions:
            lab = per_label[name]
            cells.append(_prf(lab["tp"][lbl], lab["fp"][lbl], lab["fn"][lbl])[2])
        print(f"{lbl:<11}" + "".join(f"{c:>8.4f}" for c in cells))


if __name__ == "__main__":
    main()
