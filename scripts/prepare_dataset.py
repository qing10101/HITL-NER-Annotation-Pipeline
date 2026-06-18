"""Build the 15 000-row NER test set from Amazon Reviews 2023 (McAuley Lab).

Sampling design (Design Justification.pdf):

  Edu tier  — 4 000 rows, 1 category × 4000 rows each:

  Rich tier  — 8 000 rows, 4 categories × 2000 rows each:
      Baby_Products, Toys_and_Games, Clothing_Shoes_and_Jewelry,
      Beauty_and_Personal_Care, Office_Products

  Diversity tier — 3 000 rows, 4 categories × 750 rows each:
      Pet_Supplies, Books, Automotive, Unknown

Each category JSONL is streamed directly from HuggingFace without a full
download.  Reservoir sampling keeps memory bounded regardless of file size.
The scan window is set to ≥20× the target so the reservoir is well-mixed.

Fault tolerance:
  - Each completed category is checkpointed to <out>.ckpt/<category>.jsonl.
  - On re-run the script skips any category whose checkpoint already exists,
    so a mid-run crash only loses the category that was in progress.
  - Pass --no-resume to ignore existing checkpoints and start fresh.

Network robustness:
  - Up to 3 connection attempts per category, with exponential back-off
    (5 s → 15 s → 45 s) to survive transient SSL/TCP timeouts.

Output JSONL schema:
  {"id": "<asin>_<seq>", "text": "...", "category": "...", "tier": "..."}

Usage:
  python scripts/prepare_dataset.py
  python scripts/prepare_dataset.py --out data/my_set.jsonl --seed 99
  python scripts/prepare_dataset.py --verified-only --min-len 50
  python scripts/prepare_dataset.py --no-resume   # discard checkpoints, restart
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

HF_BASE = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
    "/resolve/main/raw/review_categories"
)

DEFAULT_OUT = "data/test_set_15k.jsonl"

# Retry policy for transient connection errors
_MAX_RETRIES = 3
_RETRY_DELAYS = [5, 15, 45]  # seconds between attempts


@dataclass(frozen=True)
class CategorySpec:
    name: str
    target: int
    scan: int   # max valid rows to scan (≥20× target recommended)
    tier: str


# Scan window = 25× target for edu and rich (dense, high yield expected) and
# 20× target for diversity (sparser signal, adversarial categories may
# need more headroom).
CATEGORIES: list[CategorySpec] = [
    
    # Tier for MINOR_EDU
    CategorySpec("Office_Products",            target=4000, scan=100_000, tier="edu_rich"),

    # ── Rich tier ───────────────────────────────────────────────────────────
    CategorySpec("Baby_Products",              target=2000, scan=50_000, tier="rich"),
    CategorySpec("Toys_and_Games",             target=2000, scan=50_000, tier="rich"),
    CategorySpec("Clothing_Shoes_and_Jewelry", target=2000, scan=50_000, tier="rich"),
    CategorySpec("Beauty_and_Personal_Care",   target=2000, scan=50_000, tier="rich"),

    # ── Diversity tier ──────────────────────────────────────────────────────
    CategorySpec("Pet_Supplies", target=750, scan=15000, tier="diversity-adversarial"),
    CategorySpec("Books",        target=750, scan=15000, tier="diversity-adversarial"),
    CategorySpec("Automotive",   target=750, scan=15000, tier="diversity-generic"),
    CategorySpec("Unknown",      target=750, scan=15000, tier="diversity-generic"),
]


def _stream_sample(
    spec: CategorySpec,
    *,
    min_len: int,
    max_len: int,
    verified_only: bool,
    rng: random.Random,
) -> list[dict]:
    """Reservoir-sample ``spec.target`` raw objects from the category stream.

    Retries up to _MAX_RETRIES times on connection/SSL errors before raising.
    """
    url = f"{HF_BASE}/{spec.name}.jsonl"
    print(f"[{spec.name}] streaming {url}", flush=True)

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return _do_stream(
                spec, url,
                min_len=min_len, max_len=max_len,
                verified_only=verified_only, rng=rng,
            )
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError) as exc:
            if attempt == _MAX_RETRIES:
                raise
            delay = _RETRY_DELAYS[attempt - 1]
            print(
                f"  [{spec.name}] connection error on attempt {attempt}/{_MAX_RETRIES}"
                f" ({exc}); retrying in {delay}s …",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError("unreachable")  # satisfies type checkers


def _do_stream(
    spec: CategorySpec,
    url: str,
    *,
    min_len: int,
    max_len: int,
    verified_only: bool,
    rng: random.Random,
) -> list[dict]:
    """Single streaming attempt — called by _stream_sample."""
    req = urllib.request.Request(url, headers={"User-Agent": "ner-pipeline/1.0"})
    reservoir: list[dict] = []
    seen = 0
    seen_texts: set[str] = set()

    raw_count = 0
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            if seen >= spec.scan:
                break
            raw_count += 1
            # Heartbeat on raw lines so the user sees progress even when
            # valid rows are sparse (e.g. many short reviews filtered out).
            if raw_count % 50_000 == 0:
                print(
                    f"  [{spec.name}] {raw_count:>8} raw lines read,"
                    f" {seen} valid so far …",
                    flush=True,
                )
            line = raw_line.decode("utf-8", "ignore").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = (obj.get("text") or "").strip()
            if not (min_len <= len(text) <= max_len):
                continue

            # Bias toward verified purchases when requested.
            # Missing field → treated as unverified, so we skip if flag is set.
            if verified_only and not obj.get("verified_purchase", False):
                continue

            if text in seen_texts:
                continue
            seen_texts.add(text)
            seen += 1

            # Reservoir sampling (Algorithm R)
            if len(reservoir) < spec.target:
                reservoir.append(obj)
            else:
                j = rng.randint(0, seen - 1)
                if j < spec.target:
                    reservoir[j] = obj

            if seen % 1_000 == 0:
                print(
                    f"  [{spec.name}] scanned {seen:>6}/{spec.scan} valid rows,"
                    f" reservoir={len(reservoir)}",
                    flush=True,
                )

    collected = len(reservoir)
    print(
        f"[{spec.name}] done — scanned {seen} valid rows,"
        f" collected {collected}/{spec.target}",
        flush=True,
    )
    if collected < spec.target:
        print(
            f"  WARNING: only {collected}/{spec.target} rows collected for"
            f" {spec.name}. Increase --scan or relax --min-len / --verified-only.",
            file=sys.stderr,
        )
    return reservoir


def _ckpt_path(ckpt_dir: Path, spec: CategorySpec) -> Path:
    return ckpt_dir / f"{spec.name}.jsonl"


def _save_checkpoint(
    ckpt_dir: Path,
    spec: CategorySpec,
    records: list[dict],
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(_ckpt_path(ckpt_dir, spec), "w", encoding="utf-8") as fh:
        for obj in records:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_checkpoint(ckpt_dir: Path, spec: CategorySpec) -> list[dict] | None:
    p = _ckpt_path(ckpt_dir, spec)
    if not p.exists():
        return None
    records: list[dict] = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(
        f"[{spec.name}] resuming from checkpoint ({len(records)} rows)",
        flush=True,
    )
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the 15 k-row NER test set.")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output JSONL path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--min-len", type=int, default=30,
        help="minimum review character length (default: 30)",
    )
    ap.add_argument(
        "--max-len", type=int, default=2000,
        help="maximum review character length (default: 2000)",
    )
    ap.add_argument(
        "--verified-only", action="store_true",
        help="restrict to verified purchases to raise first-person disclosure density",
    )
    ap.add_argument(
        "--no-resume", action="store_true",
        help="ignore existing checkpoints and restart from scratch",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    ckpt_dir = out_path.parent / (out_path.stem + ".ckpt")

    if args.no_resume and ckpt_dir.exists():
        import shutil
        shutil.rmtree(ckpt_dir)
        print(f"removed checkpoint dir {ckpt_dir}", flush=True)

    rng = random.Random(args.seed)
    all_records: list[tuple[dict, CategorySpec]] = []

    for spec in CATEGORIES:
        cached = _load_checkpoint(ckpt_dir, spec)
        if cached is not None:
            records = cached
        else:
            records = _stream_sample(
                spec,
                min_len=args.min_len,
                max_len=args.max_len,
                verified_only=args.verified_only,
                rng=rng,
            )
            _save_checkpoint(ckpt_dir, spec, records)

        all_records.extend((obj, spec) for obj in records)

    rng.shuffle(all_records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, (obj, spec) in enumerate(all_records):
            rid = obj.get("parent_asin") or obj.get("asin") or f"amz_{i:05d}"
            rec = {
                "id": f"{rid}_{i:05d}",
                "text": obj["text"].strip(),
                "category": spec.name,
                "tier": spec.tier,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = len(all_records)
    print(f"\nwrote {total} rows -> {out_path}\n")

    counts: Counter[str] = Counter(spec.name for _, spec in all_records)
    header = f"{'category':<35}  {'collected':>9}  {'target':>7}"
    print(header)
    print("-" * len(header))
    for spec in CATEGORIES:
        tier_tag = "(" + spec.tier + ")"
        print(
            f"{spec.name:<35}  {counts[spec.name]:>9}  {spec.target:>7}"
            f"  {tier_tag}"
        )
    print("-" * len(header))
    print(f"{'TOTAL':<35}  {total:>9}  {sum(s.target for s in CATEGORIES):>7}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
