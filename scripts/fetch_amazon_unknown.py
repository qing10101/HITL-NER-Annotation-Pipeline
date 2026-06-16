"""Sample N reviews from the Amazon-Reviews-2023 'Unknown' category.

The raw category file (raw/review_categories/Unknown.jsonl) is ~30 GB, so we
stream it over HTTP and reservoir-sample within a bounded prefix window — only a
few MB are actually downloaded. Output is JSONL ready for the pipeline:
{"id": <parent_asin>, "text": <review body>}.

Usage:
  python scripts/fetch_amazon_unknown.py --n 100 --out data/amazon_unknown_100.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import urllib.request

URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/review_categories/Unknown.jsonl"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="rows to sample")
    ap.add_argument("--out", default="data/amazon_unknown_100.jsonl")
    ap.add_argument("--scan", type=int, default=15000,
                    help="prefix window (valid rows) to sample from")
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    reservoir: list[dict] = []
    seen = 0
    seen_texts: set[str] = set()

    req = urllib.request.Request(URL, headers={"User-Agent": "ner-pipeline/1.0"})
    print(f"streaming {URL}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:  # file-like: iterates by line
            if seen >= args.scan:
                break
            line = raw.decode("utf-8", "ignore").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (obj.get("text") or "").strip()
            if not (args.min_len <= len(text) <= args.max_len):
                continue
            if text in seen_texts:
                continue
            seen_texts.add(text)
            seen += 1
            # reservoir sampling
            if len(reservoir) < args.n:
                reservoir.append(obj)
            else:
                j = rng.randint(0, seen - 1)
                if j < args.n:
                    reservoir[j] = obj
            if seen % 2000 == 0:
                print(f"  scanned {seen} valid rows, reservoir={len(reservoir)}")

    rng.shuffle(reservoir)
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, obj in enumerate(reservoir):
            rid = obj.get("parent_asin") or obj.get("asin") or f"amz_{i:04d}"
            rec = {"id": f"{rid}_{i:03d}", "text": obj["text"].strip()}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"wrote {len(reservoir)} rows -> {args.out} (scanned {seen} valid rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
