"""Export pipeline data to CSV for human annotation and HITL review.

Two modes:

  source   Convert a source JSONL file (schema: id, text, category, tier) to
           a blank annotation sheet where human annotators fill in
           ``annotated_text`` and ``reviewer_notes``.

  queue    Convert the pipeline's review_queue.csv (failed rows) to a
           side-by-side correction sheet where reviewers fix faulty annotations.
           The ``corrected_text`` and ``reviewer_notes`` columns are left blank
           for human input.

Usage:
  # Blank annotation sheet from a source JSONL
  python scripts/export_for_review.py source data/sample_2000.jsonl

  # HITL correction sheet from the pipeline review queue
  python scripts/export_for_review.py queue output/my_sample

  # Source with filters
  python scripts/export_for_review.py source data/sample_2000.jsonl \\
      --category Baby_Products --tier rich --limit 200 \\
      --out data/baby_annotation_sheet.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# ── column layouts ─────────────────────────────────────────────────────────── #

_SOURCE_HEADER = [
    "row_num",
    "id",
    "category",
    "tier",
    "text",
    "annotated_text",   # blank — human fills this in
    "reviewer_notes",   # blank — optional human comments
]

_QUEUE_HEADER = [
    "row_num",
    "row_id",
    "error_type",
    "auditor_reason",
    "raw_text",
    "faulty_annotated_text",
    "corrected_text",   # blank — human fills in the corrected annotation
    "reviewer_notes",   # blank — optional human comments
]


# ── source mode ────────────────────────────────────────────────────────────── #

def _export_source(
    jsonl_path: Path,
    out_path: Path,
    *,
    category_filter: str | None,
    tier_filter: str | None,
    limit: int | None,
) -> int:
    """Read source JSONL and write a blank annotation sheet CSV."""
    written = 0
    skipped = 0

    with (
        jsonl_path.open(encoding="utf-8") as src,
        out_path.open("w", encoding="utf-8", newline="") as dst,
    ):
        writer = csv.writer(dst)
        writer.writerow(_SOURCE_HEADER)

        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            if category_filter and obj.get("category") != category_filter:
                continue
            if tier_filter and obj.get("tier") != tier_filter:
                continue

            written += 1
            writer.writerow([
                written,
                obj.get("id", ""),
                obj.get("category", ""),
                obj.get("tier", ""),
                obj.get("text", ""),
                "",  # annotated_text — human input
                "",  # reviewer_notes — human input
            ])

            if limit and written >= limit:
                break

    if skipped:
        print(f"  skipped {skipped} malformed lines", file=sys.stderr)
    return written


# ── queue mode ─────────────────────────────────────────────────────────────── #

def _export_queue(
    output_dir: Path,
    out_path: Path,
    *,
    error_type_filter: str | None,
    limit: int | None,
) -> int:
    """Read review_queue.csv from output_dir and write a correction sheet."""
    queue_file = output_dir / "review_queue.csv"
    if not queue_file.exists():
        print(
            f"error: review_queue.csv not found in {output_dir}",
            file=sys.stderr,
        )
        return 0

    written = 0

    with (
        queue_file.open(encoding="utf-8", newline="") as src,
        out_path.open("w", encoding="utf-8", newline="") as dst,
    ):
        reader = csv.DictReader(src)
        writer = csv.writer(dst)
        writer.writerow(_QUEUE_HEADER)

        for row in reader:
            if error_type_filter and row.get("error_type") != error_type_filter:
                continue

            written += 1
            writer.writerow([
                written,
                row.get("row_id", ""),
                row.get("error_type", ""),
                row.get("auditor_reason", ""),
                row.get("raw_text", ""),
                row.get("tagged_text", ""),
                "",  # corrected_text — human input
                "",  # reviewer_notes — human input
            ])

            if limit and written >= limit:
                break

    return written


# ── CLI ────────────────────────────────────────────────────────────────────── #

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export pipeline data to CSV for human annotation / HITL review.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    # ── source sub-command ─────────────────────────────────────────────────── #
    sp = sub.add_parser("source", help="blank annotation sheet from a source JSONL")
    sp.add_argument("jsonl", type=Path, help="path to the source .jsonl file")
    sp.add_argument(
        "--out", type=Path, default=None,
        help="output CSV path (default: same name as input with .csv extension)",
    )
    sp.add_argument("--category", default=None, help="keep only this category")
    sp.add_argument("--tier", default=None, help="keep only this tier")
    sp.add_argument("--limit", type=int, default=None, help="max rows to export")

    # ── queue sub-command ──────────────────────────────────────────────────── #
    qp = sub.add_parser(
        "queue",
        help="HITL correction sheet from a pipeline output directory",
    )
    qp.add_argument(
        "output_dir", type=Path,
        help="pipeline output directory containing review_queue.csv",
    )
    qp.add_argument(
        "--out", type=Path, default=None,
        help="output CSV path (default: <output_dir>/review_queue_export.csv)",
    )
    qp.add_argument(
        "--error-type", default=None,
        help="keep only rows matching this error_type",
    )
    qp.add_argument("--limit", type=int, default=None, help="max rows to export")

    args = ap.parse_args()

    if args.mode == "source":
        jsonl_path: Path = args.jsonl
        if not jsonl_path.exists():
            print(f"error: file not found: {jsonl_path}", file=sys.stderr)
            return 1
        out_path: Path = args.out or jsonl_path.with_suffix(".csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        filters = []
        if args.category:
            filters.append(f"category={args.category}")
        if args.tier:
            filters.append(f"tier={args.tier}")
        if args.limit:
            filters.append(f"limit={args.limit}")
        if filters:
            print(f"filters: {', '.join(filters)}")

        n = _export_source(
            jsonl_path, out_path,
            category_filter=args.category,
            tier_filter=args.tier,
            limit=args.limit,
        )
        print(f"wrote {n} rows -> {out_path}")

    else:  # queue
        output_dir: Path = args.output_dir
        if not output_dir.is_dir():
            print(f"error: directory not found: {output_dir}", file=sys.stderr)
            return 1
        out_path = args.out or output_dir / "review_queue_export.csv"

        if args.error_type:
            print(f"filters: error_type={args.error_type}")

        n = _export_queue(
            output_dir, out_path,
            error_type_filter=args.error_type,
            limit=args.limit,
        )
        if n:
            print(f"wrote {n} rows -> {out_path}")
        else:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
