"""make_balanced_split.py

Build a label-balanced held-out test set from the full gold corpus, per the
advisor's spec: N rows each containing MINOR / GENDER / KINSHIP entities, plus
N rows with none of the three (default N=100 -> 400 test rows). The rest of the
corpus becomes the training pool.

Buckets are "at-least-one" (a bucket row contains that dimension, possibly
alongside others) and are kept mutually disjoint by assigning rarest-first
(GENDER -> MINOR -> KINSHIP): once a row is claimed for a bucket it is removed
from the pool, so no row appears in two buckets and there are no duplicates.
The full corpus has no duplicate raw_text, so text-level duplication is a
non-issue.

Labels are folded to the 3 coarse dimensions only to categorize rows; the rows
themselves are written out unchanged (original entities_json), because
common.load_examples folds at load time downstream.

Outputs (default --out-dir output/):
    test_balanced_400.csv       the 400-row balanced test set
    train_pool_19600.csv        everything else (source of train/dev)
    test_balanced_400_manifest.csv   row_id,bucket  (for per-bucket analysis)

Usage:
    python benchmark/supervised/make_balanced_split.py \
        --gold output/gold_standard_merged.csv --per-bucket 100

Then wire it into the trainers' data prep:
    python benchmark/supervised/prepare_data.py \
        --gold output/train_pool_19600.csv \
        --test-gold output/test_balanced_400.csv --dev-frac 0.1
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

# Fold the 5 fine gold labels onto the 3 coarse dimensions (mirrors
# common.COARSE_3DIM / benchmark/llm/evaluate.py). FAM_KIN -> KINSHIP is 1:1.
COARSE_3DIM = {
    "MINOR_AGE": "MINOR", "MINOR_EDU": "MINOR",
    "GEN_NOUN": "GENDER", "GEN_PHYS": "GENDER",
    "FAM_KIN": "KINSHIP",
}

# Rarest-first assignment order (GENDER is the scarcest dimension corpus-wide,
# so it claims its rows before the commoner dimensions can absorb them).
BUCKET_ORDER = ["GENDER", "MINOR", "KINSHIP"]


def row_dims(entities_json: str) -> set[str]:
    """The set of coarse dimensions present in a row's entities."""
    ents = json.loads(entities_json or "[]")
    return {COARSE_3DIM.get(e["label"], e["label"]) for e in ents}


def build(gold_path: Path, out_dir: Path, per_bucket: int, seed: int) -> None:
    with open(gold_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    dims = [row_dims(r["entities_json"]) for r in rows]
    print(f"Loaded {len(rows)} rows from {gold_path}")

    texts = Counter(r["raw_text"] for r in rows)
    dup = sum(c for c in texts.values() if c > 1)
    if dup:
        print(f"WARNING: {dup} rows share a raw_text with another row.")

    rng = random.Random(seed)
    assigned: dict[int, str] = {}

    # Rarest-first: claim each dimension's rows from what's still unassigned.
    for label in BUCKET_ORDER:
        candidates = [i for i in range(len(rows))
                      if i not in assigned and label in dims[i]]
        if len(candidates) < per_bucket:
            raise SystemExit(
                f"Only {len(candidates)} unclaimed rows contain {label}, "
                f"need {per_bucket}. Lower --per-bucket.")
        rng.shuffle(candidates)
        for i in candidates[:per_bucket]:
            assigned[i] = label

    # "none" bucket: rows with none of the three dimensions.
    none_candidates = [i for i in range(len(rows))
                       if i not in assigned and not (dims[i] & set(BUCKET_ORDER))]
    if len(none_candidates) < per_bucket:
        raise SystemExit(f"Only {len(none_candidates)} empty rows, need {per_bucket}.")
    rng.shuffle(none_candidates)
    for i in none_candidates[:per_bucket]:
        assigned[i] = "NONE"

    test_idx = sorted(assigned)
    train_idx = [i for i in range(len(rows)) if i not in assigned]

    out_dir.mkdir(parents=True, exist_ok=True)
    test_path = out_dir / "test_balanced_400.csv"
    train_path = out_dir / "train_pool_19600.csv"
    manifest_path = out_dir / "test_balanced_400_manifest.csv"

    def write_rows(path: Path, idxs: list[int]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for i in idxs:
                w.writerow(rows[i])

    write_rows(test_path, test_idx)
    write_rows(train_path, train_idx)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "bucket"])
        for i in test_idx:
            w.writerow([rows[i]["row_id"], assigned[i]])

    # Report: bucket sizes and how much each bucket's rows co-occur with the
    # other dimensions (the "bleed" inherent to at-least-one buckets).
    print(f"\nTest set: {len(test_idx)} rows -> {test_path}")
    per_bucket_rows: dict[str, list[int]] = {}
    for i in test_idx:
        per_bucket_rows.setdefault(assigned[i], []).append(i)
    for b in BUCKET_ORDER + ["NONE"]:
        idxs = per_bucket_rows.get(b, [])
        others = Counter()
        for i in idxs:
            for d in dims[i]:
                if d != b:
                    others[d] += 1
        extra = f" also-contains={dict(others)}" if others else ""
        print(f"  {b:8s} rows={len(idxs)}{extra}")
    print(f"\nTrain pool: {len(train_idx)} rows -> {train_path}")
    print(f"Manifest (row_id,bucket): {manifest_path}")


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--gold", default=str(here.parent.parent / "output" / "gold_standard_merged.csv"))
    ap.add_argument("--out-dir", default=str(here.parent.parent / "output"))
    ap.add_argument("--per-bucket", type=int, default=100)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    build(Path(args.gold), Path(args.out_dir), args.per_bucket, args.seed)


if __name__ == "__main__":
    main()
