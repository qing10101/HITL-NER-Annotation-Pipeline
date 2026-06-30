"""Merge human-reviewed queue rows into the gold standard dataset.

Reads ``output/human_reviewed.csv``, parses each row's ``human_annotation``
column through the deterministic regex parser to compute span offsets, then
appends those rows to ``output/gold_standard.csv`` and writes the combined
result to ``output/gold_standard_merged.csv``.

Rows with an empty ``human_annotation`` (not yet adjudicated) are skipped.
Rows whose ``human_annotation`` fails tag parsing are reported and skipped.
Duplicate row_ids already present in gold_standard.csv are also skipped.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.parser import TagParseError, parse_tagged_text  # noqa: E402

HUMAN_REVIEWED = ROOT / "output" / "human_reviewed.csv"
GOLD_STANDARD = ROOT / "output" / "gold_standard.csv"
OUTPUT = ROOT / "output" / "gold_standard_merged.csv"

GOLD_FIELDNAMES = ["row_id", "raw_text", "tagged_text", "num_entities", "entities_json"]


def main() -> None:
    # Load existing gold standard rows and track which IDs are already present.
    gold_rows: list[dict] = []
    existing_ids: set[str] = set()
    with GOLD_STANDARD.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gold_rows.append(row)
            existing_ids.add(row["row_id"])

    # Process human-reviewed rows.
    added = skipped_empty = skipped_duplicate = skipped_parse_error = 0
    with HUMAN_REVIEWED.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = row["row_id"]
            human_annotation = row.get("human_annotation", "").strip()

            if not human_annotation:
                skipped_empty += 1
                continue

            if row_id in existing_ids:
                print(f"  [skip duplicate] {row_id}")
                skipped_duplicate += 1
                continue

            try:
                _, spans = parse_tagged_text(human_annotation)
            except TagParseError as exc:
                print(f"  [parse error] {row_id}: {exc}")
                skipped_parse_error += 1
                continue

            entities = [
                {"label": s.label, "text": s.text, "start": s.start, "end": s.end}
                for s in spans
            ]
            gold_rows.append({
                "row_id": row_id,
                "raw_text": row["raw_text"],
                "tagged_text": human_annotation,
                "num_entities": len(entities),
                "entities_json": json.dumps(entities),
            })
            existing_ids.add(row_id)
            added += 1

    # Write merged output.
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GOLD_FIELDNAMES)
        writer.writeheader()
        writer.writerows(gold_rows)

    print(
        f"\nDone. {added} row(s) added, {skipped_empty} skipped (empty annotation), "
        f"{skipped_duplicate} skipped (duplicate), {skipped_parse_error} skipped (parse error)."
    )
    print(f"Output: {OUTPUT}")
    print(f"Total rows in merged file: {len(gold_rows)}")


if __name__ == "__main__":
    main()
