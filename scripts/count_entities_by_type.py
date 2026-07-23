"""Count entities per type in gold_standard_merged.csv.

Reads the entities_json column of each row and tallies how many entity
spans occur per label, printing a summary table sorted by count.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
GOLD_CSV = ROOT / "output" / "gold_standard_merged.csv"


def count_entities(csv_path: Path) -> Counter:
    counts: Counter = Counter()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("entities_json") or "[]"
            entities = json.loads(raw)
            for ent in entities:
                counts[ent["label"]] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=GOLD_CSV,
        type=Path,
        help="Path to gold standard CSV (default: output/gold_standard_merged.csv)",
    )
    args = parser.parse_args()

    counts = count_entities(args.csv_path)
    total = sum(counts.values())

    width = max((len(label) for label in counts), default=5)
    for label, count in counts.most_common():
        print(f"{label:<{width}}  {count}")
    print(f"{'TOTAL':<{width}}  {total}")


if __name__ == "__main__":
    main()
