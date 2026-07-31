"""prepare_data.py

Builds the single shared train/dev/test split every trainer consumes, plus
the per-model input formats. Stdlib-only — run it before installing any of
the ML frameworks.

Outputs under --out-dir (default: benchmark/supervised/data):
    splits/{train,dev,test}.csv   gold-schema CSVs (source of truth per split)
    bio/{train,dev,test}.jsonl    tokens + BIO tags + char offsets
                                  (BiLSTM-CRF and SpanMarker)

Usage:
    python benchmark/supervised/prepare_data.py \
        --gold benchmark/gold_standard_merged.csv \
        --neg-ratio 2.0
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import common


def downsample_negatives(
    examples: list[common.Example], neg_ratio: float, seed: int
) -> list[common.Example]:
    """Keep at most neg_ratio empty rows per entity-bearing row (train only)."""
    positives = [ex for ex in examples if ex.entities]
    negatives = [ex for ex in examples if not ex.entities]
    keep = int(neg_ratio * len(positives))
    if len(negatives) > keep:
        rng = random.Random(seed)
        negatives = rng.sample(negatives, keep)
    kept = positives + negatives
    kept.sort(key=lambda ex: ex.row_id)
    return kept


def write_bio_jsonl(path: Path, examples: list[common.Example]) -> tuple[int, int]:
    """Write tokens/tags/offsets per row; returns (n_adjusted, n_dropped)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    total_adjusted = total_dropped = 0
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            tokens = common.tokenize(ex.text)
            if not tokens:
                continue
            tags, adjusted, dropped = common.spans_to_bio(tokens, ex.entities)
            total_adjusted += adjusted
            total_dropped += dropped
            f.write(json.dumps({
                "row_id": ex.row_id,
                "text": ex.text,
                "tokens": [t for t, _, _ in tokens],
                "tags": tags,
                "starts": [s for _, s, _ in tokens],
                "ends": [e for _, _, e in tokens],
            }, ensure_ascii=False) + "\n")
    return total_adjusted, total_dropped


def label_counts(examples: list[common.Example]) -> Counter:
    return Counter(e["label"] for ex in examples for e in ex.entities)


def prepare(
    gold_path: str | Path,
    out_dir: str | Path,
    seed: int = 13,
    neg_ratio: float | None = None,
    test_gold: str | Path | None = None,
    dev_frac: float = 0.1,
) -> dict[str, list[common.Example]]:
    out_dir = Path(out_dir)
    examples = common.load_examples(gold_path)
    print(f"Loaded {len(examples)} rows from {gold_path}")

    if test_gold is not None:
        # Fixed held-out test set (e.g. the 500 human-labeled rows). The --gold
        # pool is split into train/dev only; test comes verbatim from test_gold.
        splits = common.split_examples(
            examples, train_frac=1.0 - dev_frac, dev_frac=dev_frac, seed=seed,
        )
        assert not splits["test"], "test slice should be empty when test_gold is given"
        splits["test"] = common.load_examples(test_gold)
        print(f"Loaded {len(splits['test'])} held-out test rows from {test_gold}")
    else:
        splits = common.split_examples(examples, seed=seed)

    if neg_ratio is not None:
        before = len(splits["train"])
        splits["train"] = downsample_negatives(splits["train"], neg_ratio, seed)
        print(f"Downsampled train negatives: {before} -> {len(splits['train'])} rows "
              f"(neg_ratio={neg_ratio})")

    for name, exs in splits.items():
        common.write_examples_csv(out_dir / "splits" / f"{name}.csv", exs)
        adjusted, dropped = write_bio_jsonl(out_dir / "bio" / f"{name}.jsonl", exs)
        counts = label_counts(exs)
        n_pos = sum(1 for ex in exs if ex.entities)
        print(f"[{name}] rows={len(exs)} with_entities={n_pos} "
              f"entities={dict(counts.most_common())} "
              f"bio_adjusted={adjusted} bio_dropped={dropped}")

    return splits


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--gold", default=str(here.parent / "gold_standard_merged.csv"))
    ap.add_argument("--out-dir", default=str(here / "data"))
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--neg-ratio", type=float, default=None,
        help="If set, keep at most this many no-entity rows per entity-bearing "
             "row in TRAIN (dev/test always keep everything).",
    )
    ap.add_argument(
        "--test-gold", default=None,
        help="Path to a fixed held-out test CSV (e.g. the 500 human-labeled rows). "
             "When given, --gold is split into train/dev only and this file becomes "
             "the test split.",
    )
    ap.add_argument(
        "--dev-frac", type=float, default=0.1,
        help="Fraction of the --gold pool held out for dev early-stopping when "
             "--test-gold is used (default 0.1).",
    )
    args = ap.parse_args()
    prepare(args.gold, args.out_dir, seed=args.seed, neg_ratio=args.neg_ratio,
            test_gold=args.test_gold, dev_frac=args.dev_frac)


if __name__ == "__main__":
    main()
