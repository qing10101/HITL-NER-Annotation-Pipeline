"""Compare human-annotated sample against the pipeline gold standard.

Joins sample_500_human.csv and gold_standard_merged.csv on row_id, parses
each annotated text into spans, then produces two levels of diff:

Row-level  (output/diff_human_vs_gold.csv)
  Status per row: MATCH | DIFF | PARSE_ERROR | ONLY_IN_HUMAN | ONLY_IN_GOLD.
  Span equality at this level is (label, text) — position-only differences
  still count as a match so the row isn't penalised for a pure offset shift.

Span-level  (output/span_level_diff.csv)
  One row per non-exact span, classified as:
    BOUNDARY_SHIFT  — same (label, text), different (start, end)
    LABEL_CONFLICT  — same (text, start, end), different label
    HUMAN_ONLY      — span present in human but absent from gold
    GOLD_ONLY       — span present in gold but absent from human
  Plus a per-label summary table printed to stdout.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.parser import TagParseError, parse_tagged_text  # noqa: E402

HUMAN_CSV      = ROOT / "output" / "sample_500_human.csv"
GOLD_CSV       = ROOT / "output" / "gold_standard_merged.csv"
ROW_DIFF_CSV   = ROOT / "output" / "diff_human_vs_gold.csv"
SPAN_DIFF_CSV  = ROOT / "output" / "span_level_diff.csv"

LABELS = ["MINOR_AGE", "MINOR_EDU", "GEN_NOUN", "GEN_PHYS", "FAM_KIN"]

ROW_FIELDS = [
    "row_id",
    "status",           # MATCH | DIFF | PARSE_ERROR | ONLY_IN_HUMAN | ONLY_IN_GOLD
    "human_entities",   # JSON [(label, text), ...]
    "gold_entities",
    "only_in_human",
    "only_in_gold",
]

SPAN_FIELDS = [
    "row_id",
    "diff_type",        # BOUNDARY_SHIFT | LABEL_CONFLICT | HUMAN_ONLY | GOLD_ONLY
    "label",
    "human_text",
    "gold_text",
    "human_offsets",    # "start:end" or ""
    "gold_offsets",
]

# (label, text, start, end)
SpanTuple = tuple[str, str, int, int]


def _parse(tagged_text: str) -> Optional[list[SpanTuple]]:
    """Return full span tuples or None on parse failure."""
    if not tagged_text or not tagged_text.strip():
        return []
    try:
        _, spans = parse_tagged_text(tagged_text)
    except TagParseError:
        return None
    return [(s.label, s.text, s.start, s.end) for s in spans]


def _lt(spans: list[SpanTuple]) -> list[tuple[str, str]]:
    """Reduce full tuples to (label, text) pairs for row-level comparison."""
    return sorted((lbl, txt) for lbl, txt, _, _ in spans)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--human-csv", type=Path, default=HUMAN_CSV,
        help=f"Human-annotated CSV with row_id/annotated_text columns (default: {HUMAN_CSV})",
    )
    p.add_argument(
        "--gold-csv", type=Path, default=GOLD_CSV,
        help=f"Gold standard CSV with row_id/tagged_text columns (default: {GOLD_CSV})",
    )
    p.add_argument(
        "--row-diff-out", type=Path, default=ROW_DIFF_CSV,
        help=f"Path to write the row-level diff CSV (default: {ROW_DIFF_CSV})",
    )
    p.add_argument(
        "--span-diff-out", type=Path, default=SPAN_DIFF_CSV,
        help=f"Path to write the span-level diff CSV (default: {SPAN_DIFF_CSV})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load annotations ────────────────────────────────────────────────────
    human_text: dict[str, str] = {}
    with args.human_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            human_text[row["row_id"]] = row["annotated_text"]

    gold_text: dict[str, str] = {}
    with args.gold_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gold_text[row["row_id"]] = row["tagged_text"]

    # Parse once; reuse for both levels.
    human_spans: dict[str, Optional[list[SpanTuple]]] = {
        rid: _parse(txt) for rid, txt in human_text.items()
    }
    gold_spans: dict[str, Optional[list[SpanTuple]]] = {
        rid: _parse(txt) for rid, txt in gold_text.items()
    }

    all_ids = sorted(set(human_text) | set(gold_text))

    # ── Row-level diff ───────────────────────────────────────────────────────
    row_status: Counter[str] = Counter()
    row_rows: list[dict] = []

    for row_id in all_ids:
        in_h = row_id in human_spans
        in_g = row_id in gold_spans

        if in_h and not in_g:
            row_status["ONLY_IN_HUMAN"] += 1
            h_lt = _lt(human_spans[row_id] or [])
            row_rows.append({
                "row_id": row_id, "status": "ONLY_IN_HUMAN",
                "human_entities": json.dumps(h_lt), "gold_entities": "[]",
                "only_in_human": json.dumps(h_lt), "only_in_gold": "[]",
            })
            continue

        if in_g and not in_h:
            row_status["ONLY_IN_GOLD"] += 1
            g_lt = _lt(gold_spans[row_id] or [])
            row_rows.append({
                "row_id": row_id, "status": "ONLY_IN_GOLD",
                "human_entities": "[]", "gold_entities": json.dumps(g_lt),
                "only_in_human": "[]", "only_in_gold": json.dumps(g_lt),
            })
            continue

        h = human_spans[row_id]
        g = gold_spans[row_id]

        if h is None or g is None:
            status = "PARSE_ERROR"
            only_h: list = []
            only_g: list = []
        else:
            h_set = set(_lt(h))
            g_set = set(_lt(g))
            only_h = sorted(h_set - g_set)
            only_g = sorted(g_set - h_set)
            status = "MATCH" if not only_h and not only_g else "DIFF"

        row_status[status] += 1
        row_rows.append({
            "row_id": row_id, "status": status,
            "human_entities": json.dumps(_lt(h) if h is not None else []),
            "gold_entities":  json.dumps(_lt(g) if g is not None else []),
            "only_in_human":  json.dumps(only_h),
            "only_in_gold":   json.dumps(only_g),
        })

    with args.row_diff_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(row_rows)

    # ── Span-level diff ──────────────────────────────────────────────────────
    shared_ids = [rid for rid in all_ids
                  if rid in human_spans and rid in gold_spans
                  and human_spans[rid] is not None and gold_spans[rid] is not None]

    span_rows: list[dict] = []
    total_h = total_g = exact = boundary = conflict = h_only = g_only = 0
    per_h:  Counter[str] = Counter()
    per_g:  Counter[str] = Counter()
    per_ex: Counter[str] = Counter()
    per_bd: Counter[str] = Counter()
    per_ho: Counter[str] = Counter()
    per_go: Counter[str] = Counter()

    for row_id in shared_ids:
        h = human_spans[row_id]   # type: ignore[index]
        g = gold_spans[row_id]    # type: ignore[index]

        total_h += len(h)
        total_g += len(g)
        for lbl, _, _, _ in h: per_h[lbl] += 1
        for lbl, _, _, _ in g: per_g[lbl] += 1

        h_set = set(h)
        g_set = set(g)

        matched = h_set & g_set
        exact += len(matched)
        for lbl, _, _, _ in matched: per_ex[lbl] += 1

        h_rem = [sp for sp in h if sp not in matched]
        g_rem = [sp for sp in g if sp not in matched]

        # Boundary shifts: same (label, text), different (start, end)
        h_lt_map = {(lbl, txt): (s, e) for lbl, txt, s, e in h_rem}
        g_lt_map = {(lbl, txt): (s, e) for lbl, txt, s, e in g_rem}
        for key in set(h_lt_map) & set(g_lt_map):
            lbl, txt = key
            boundary += 1
            per_bd[lbl] += 1
            span_rows.append({
                "row_id": row_id, "diff_type": "BOUNDARY_SHIFT", "label": lbl,
                "human_text": txt, "gold_text": txt,
                "human_offsets": "{}:{}".format(*h_lt_map[key]),
                "gold_offsets":  "{}:{}".format(*g_lt_map[key]),
            })

        h_rem2 = [(lbl, txt, s, e) for lbl, txt, s, e in h_rem if (lbl, txt) not in g_lt_map]
        g_rem2 = [(lbl, txt, s, e) for lbl, txt, s, e in g_rem if (lbl, txt) not in h_lt_map]

        # Label conflicts: same (text, start, end), different label
        h_pos = {(txt, s, e): lbl for lbl, txt, s, e in h_rem2}
        g_pos = {(txt, s, e): lbl for lbl, txt, s, e in g_rem2}
        for pos_key in set(h_pos) & set(g_pos):
            if h_pos[pos_key] != g_pos[pos_key]:
                conflict += 1
                txt, s, e = pos_key
                span_rows.append({
                    "row_id": row_id, "diff_type": "LABEL_CONFLICT",
                    "label": f"{h_pos[pos_key]}→{g_pos[pos_key]}",
                    "human_text": txt, "gold_text": txt,
                    "human_offsets": f"{s}:{e}", "gold_offsets": f"{s}:{e}",
                })

        h_rem3 = [(lbl, txt, s, e) for lbl, txt, s, e in h_rem2 if (txt, s, e) not in g_pos]
        g_rem3 = [(lbl, txt, s, e) for lbl, txt, s, e in g_rem2 if (txt, s, e) not in h_pos]

        for lbl, txt, s, e in h_rem3:
            h_only += 1
            per_ho[lbl] += 1
            span_rows.append({
                "row_id": row_id, "diff_type": "HUMAN_ONLY", "label": lbl,
                "human_text": txt, "gold_text": "",
                "human_offsets": f"{s}:{e}", "gold_offsets": "",
            })
        for lbl, txt, s, e in g_rem3:
            g_only += 1
            per_go[lbl] += 1
            span_rows.append({
                "row_id": row_id, "diff_type": "GOLD_ONLY", "label": lbl,
                "human_text": "", "gold_text": txt,
                "human_offsets": "", "gold_offsets": f"{s}:{e}",
            })

    with args.span_diff_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SPAN_FIELDS)
        writer.writeheader()
        writer.writerows(span_rows)

    # ── Print summaries ──────────────────────────────────────────────────────
    shared = len(set(human_text) & set(gold_text))
    parse_errors = sum(
        1 for rid in set(human_text) & set(gold_text)
        if human_spans[rid] is None or gold_spans[rid] is None
    )

    print("=== ROW-LEVEL DIFF ===\n")
    print(f"Rows in human CSV       : {len(human_text)}")
    print(f"Rows in gold standard   : {len(gold_text)}")
    print(f"Rows in both (shared)   : {shared}")
    print()
    print(f"  MATCH                 : {row_status['MATCH']}")
    print(f"  DIFF                  : {row_status['DIFF']}")
    print(f"  PARSE_ERROR           : {row_status.get('PARSE_ERROR', 0)}")
    print(f"  ONLY_IN_HUMAN         : {row_status['ONLY_IN_HUMAN']}")
    print(f"  ONLY_IN_GOLD          : {row_status['ONLY_IN_GOLD']}")
    print()
    if row_status["DIFF"]:
        print("Differing rows:")
        for r in row_rows:
            if r["status"] != "DIFF":
                continue
            print(f"  {r['row_id']}")
            for lbl, txt in json.loads(r["only_in_human"]):
                print(f"    - human only  [{lbl}] \"{txt}\"")
            for lbl, txt in json.loads(r["only_in_gold"]):
                print(f"    + gold only   [{lbl}] \"{txt}\"")
    print(f"\nWritten: {args.row_diff_out}")

    print("\n=== SPAN-LEVEL DIFF ===\n")
    print(f"Shared rows compared    : {len(shared_ids)}  ({parse_errors} skipped — parse error)")
    print(f"Total human spans       : {total_h}")
    print(f"Total gold spans        : {total_g}")
    print()
    agree_h = f"{exact/total_h*100:.1f}%" if total_h else "n/a"
    agree_g = f"{exact/total_g*100:.1f}%" if total_g else "n/a"
    print(f"Exact matches           : {exact}")
    print(f"Boundary shifts         : {boundary}  (same label+text, diff offsets)")
    print(f"Label conflicts         : {conflict}  (same position, diff label)")
    print(f"Human-only spans        : {h_only}")
    print(f"Gold-only spans         : {g_only}")
    print(f"Agreement (human basis) : {agree_h}")
    print(f"Agreement (gold basis)  : {agree_g}")
    print()
    print(f"{'Label':<14} {'Human':>6} {'Gold':>6} {'Match':>6} {'BndSft':>7} {'H-Only':>7} {'G-Only':>7}")
    print("-" * 58)
    for lbl in LABELS:
        print(f"{lbl:<14} {per_h[lbl]:>6} {per_g[lbl]:>6} {per_ex[lbl]:>6} "
              f"{per_bd[lbl]:>7} {per_ho[lbl]:>7} {per_go[lbl]:>7}")
    print(f"\nWritten: {args.span_diff_out}")


if __name__ == "__main__":
    main()
