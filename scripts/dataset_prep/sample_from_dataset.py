"""Randomly sample N rows from the 10 k test set.

Usage:
  python scripts/sample_from_dataset.py --n 500
  python scripts/sample_from_dataset.py --n 200 --seed 99 --out data/my_sample.jsonl
  python scripts/sample_from_dataset.py --n 100 --src data/my_set.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEFAULT_SRC = "data/test_set_180k.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description="Random sample from the NER dataset.")
    ap.add_argument("--n", type=int, required=True, help="number of rows to sample")
    ap.add_argument("--src", default=DEFAULT_SRC, help="source JSONL file")
    ap.add_argument("--out", default=None, help="output path (default: data/sample_<n>.jsonl)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"error: source file not found: {src}")
        return 1

    rows = src.read_text(encoding="utf-8").splitlines()
    rows = [r for r in rows if r.strip()]

    if args.n > len(rows):
        print(f"error: --n {args.n} exceeds available rows ({len(rows)})")
        return 1

    sample = random.Random(args.seed).sample(rows, args.n)

    out = Path(args.out) if args.out else src.parent / f"sample_{args.n}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sample) + "\n", encoding="utf-8")

    print(f"sampled {args.n}/{len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
