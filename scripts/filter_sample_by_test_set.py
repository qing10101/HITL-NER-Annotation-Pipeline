"""
Filter out rows from sample_2000.csv that appear in test_set_180k_minor_edu.csv,
matching on the `id` column. Writes the remaining rows to a new CSV.
"""

import argparse
import csv
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

DEFAULT_SAMPLE = DATA_DIR / "sample_2000.csv"
DEFAULT_TEST_SET = DATA_DIR / "test_set_180k_minor_edu.csv"
DEFAULT_OUTPUT = DATA_DIR / "sample_filtered.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default=DEFAULT_SAMPLE, type=Path)
    parser.add_argument("--test-set", default=DEFAULT_TEST_SET, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    args = parser.parse_args()

    with open(args.test_set, newline="", encoding="utf-8") as f:
        test_ids = {row["id"] for row in csv.DictReader(f)}

    kept, removed = 0, 0
    with open(args.sample, newline="", encoding="utf-8") as fin, \
         open(args.output, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["id"] in test_ids:
                removed += 1
            else:
                writer.writerow(row)
                kept += 1

    print(f"Test-set IDs loaded : {len(test_ids)}")
    print(f"Rows removed        : {removed}")
    print(f"Rows kept           : {kept}")
    print(f"Output written to   : {args.output}")


if __name__ == "__main__":
    main()
