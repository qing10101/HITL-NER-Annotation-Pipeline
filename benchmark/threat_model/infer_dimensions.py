"""infer_dimensions.py

Threat-model dimension attack: for each review of the balanced 400 set, can an
LLM attacker tell whether it discloses each sensitive dimension (MINOR,
GENDER, KINSHIP), on the original text and on the redacted text?

  original  the raw review text
  redacted  every tagged span replaced by [REDACTED]; the attacker judges
            whether the review, hidden segments included, disclosed each
            dimension

The attacker knows the dimension definitions (a codebook-aware adversary) and
returns, per dimension, a yes/no verdict, a confidence and a short reason.
A row's gold verdict for a dimension is yes when it has at least one
human-validated span of that dimension (MINOR_AGE/MINOR_EDU -> MINOR,
GEN_NOUN/GEN_PHYS -> GENDER, FAM_KIN -> KINSHIP).

By default only the 300 rows with entities (MINOR, GENDER and KINSHIP buckets)
are attacked; a row's negatives are the dimensions it has no span of. Pass
--include-none to add the 100 NONE-bucket rows, negatives for all three.

Output: one row per (review, condition) in --out-csv. Resumable: reviews
already attacked are skipped; reviews that errored are retried on the next run.

Usage:
    python benchmark/threat_model/infer_dimensions.py run --concurrency 8
    python benchmark/threat_model/infer_dimensions.py score
    python benchmark/threat_model/infer_dimensions.py run --condition redacted   # one condition only
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import infer_profile as ip  # noqa: E402
from pipeline.config import CONFIG  # noqa: E402

OUT_CSV = ip.RESULTS_DIR / "dimension_inference.csv"
CONDITIONS = ["original", "redacted"]
DIMENSIONS = ["minor", "gender", "kinship"]
DIMENSION = {"MINOR_AGE": "minor", "MINOR_EDU": "minor", "GEN_NOUN": "gender",
             "GEN_PHYS": "gender", "FAM_KIN": "kinship"}

_HEADER = ["row_id", "bucket", "condition",
           *(f"gold_{d}" for d in DIMENSIONS),
           *(f"{d}_{k}" for d in DIMENSIONS for k in ("present", "confidence", "reason")),
           "error"]


class Verdict(BaseModel):
    present: bool
    confidence: Literal["low", "medium", "high"]
    reason: str


class DimensionInference(BaseModel):
    minor: Verdict
    gender: Verdict
    kinship: Verdict


SYSTEM_PROMPT = """\
You are a privacy attacker analysing a product review. Decide, for each of three
categories, whether the review discloses that kind of personal information about a
real person in the reviewer's life.

Categories:
- minor: the age, developmental stage, or school stage of a real child under 18 in the
  reviewer's life (e.g. "my 5 year old", "our toddler", "my son in 2nd grade", "my
  little one").
- gender: the gender of the reviewer or their romantic partner, revealed by a gendered
  noun (e.g. "my husband", "as a mom") or a sex-specific body or experience (e.g.
  "my pregnancy", "while breastfeeding", "my prostate").
- kinship: a family relationship term naming one of the reviewer's relatives (e.g.
  "my son", "our daughter", "the grandkids", "my brother", "my mother-in-law").

Some segments of the review may be hidden and shown as [REDACTED]. Judge the review as
it was originally written, hidden segments included: infer from the surrounding
context what each hidden segment most likely said, exactly as an attacker would, using
grammar, pronouns, products, ages and roles mentioned elsewhere.

For each category return:
- present: true if the review (as originally written) discloses it, otherwise false.
- confidence: low, medium, or high.
- reason: one short sentence on the cues you used.
"""


def _rows(include_none: bool) -> list[dict]:
    """Every review of the balanced 400 set, with gold verdicts and both attacker views."""
    from pipeline.parser import parse_and_verify

    buckets = {r["row_id"]: r["bucket"] for r in ip._read(ip.MANIFEST_CSV)}
    out = []
    for r in ip._read(ip.TAGGED_CSV):
        if buckets.get(r["row_id"]) == "NONE" and not include_none:
            continue
        raw = r["raw_text"]
        spans = parse_and_verify(r["tagged_text"], raw)
        redacted = raw
        for s in sorted(spans, key=lambda s: s.start, reverse=True):
            redacted = redacted[:s.start] + ip.REDACTION + redacted[s.end:]
        present = {DIMENSION[s.label] for s in spans}
        out.append({"row_id": r["row_id"], "bucket": buckets.get(r["row_id"], ""),
                    "gold": {d: d in present for d in DIMENSIONS},
                    "views": {"original": raw, "redacted": redacted}})
    return out


def _load_done(path: Path) -> dict[tuple[str, str], dict]:
    """Latest result per (row_id, condition) (later lines override earlier retries)."""
    if not path.exists():
        return {}
    return {(r["row_id"], r["condition"]): r for r in ip._read(path)}


# ── run ──────────────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> None:
    from pipeline.providers import build_provider

    CONFIG.require_keys([args.model])
    provider = build_provider(args.model)

    conditions = CONDITIONS if args.condition == "both" else [args.condition]
    rows = _rows(args.include_none)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not args.out_csv.exists():
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_HEADER)
    done = {key for key, r in _load_done(args.out_csv).items() if not r["error"]}
    todo = [(r, c) for r in rows for c in conditions if (r["row_id"], c) not in done]
    print(f"Attacker: {args.model} (temperature={args.temperature})")
    print(f"Rows: {len(rows)}  conditions: {', '.join(conditions)}  to attack: {len(todo)}")

    sem = asyncio.Semaphore(args.concurrency)
    stats: Counter[str] = Counter()

    async def _one(row: dict, cond: str) -> None:
        async with sem:
            res, error = None, ""
            try:
                res = await provider.generate_structured_async(
                    SYSTEM_PROMPT, f"REVIEW:\n{row['views'][cond]}", args.temperature, DimensionInference)
                if not isinstance(res, DimensionInference):
                    res, error = None, "No parseable structured output."
            except Exception as exc:  # noqa: BLE001 - isolate per-row failures
                error = f"{type(exc).__name__}: {exc}"
        if res:
            values = [x for d in DIMENSIONS
                      for v in [getattr(res, d)] for x in (v.present, v.confidence, v.reason)]
        else:
            values = [""] * (3 * len(DIMENSIONS))
        with args.out_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row["row_id"], row["bucket"], cond,
                                    *(row["gold"][d] for d in DIMENSIONS), *values, error])
        stats["error" if error else "ok"] += 1
        n = sum(stats.values())
        if n % 50 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}  " + "  ".join(f"{k}={v}" for k, v in sorted(stats.items())), flush=True)

    await asyncio.gather(*(_one(r, c) for r, c in todo))
    if stats["error"]:
        print(f"\n{stats['error']} row(s) errored; rerun `run` to retry them.")
    print(f"Written: {args.out_csv}")


# ── score ────────────────────────────────────────────────────────────────────

def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def _score(args: argparse.Namespace) -> None:
    results = [r for r in _load_done(args.out_csv).values() if not r["error"]]
    if not results:
        sys.exit(f"No results in {args.out_csv}; run `run` first.")

    for cond in CONDITIONS:
        rows = [r for r in results if r["condition"] == cond]
        if not rows:
            continue
        print(f"== {cond}: {len(rows)} reviews\n")
        header = (f"  {'Dimension':<10}{'Pos':>5}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}"
                  f"{'Acc':>8}{'P':>8}{'R':>8}{'F1':>8}")
        print(header)
        for d in DIMENSIONS:
            c = Counter((r[f"gold_{d}"] == "True", r[f"{d}_present"] == "True") for r in rows)
            tp, fp, fn, tn = c[(True, True)], c[(False, True)], c[(True, False)], c[(False, False)]
            p, rc, f = _prf(tp, fp, fn)
            print(f"  {d:<10}{tp + fn:>5}{tp:>5}{fp:>5}{fn:>5}{tn:>5}"
                  f"{(tp + tn) / len(rows):>8.1%}{p:>8.3f}{rc:>8.3f}{f:>8.3f}")

        print(f"\n  Recall by bucket (gold-positive rows flagged present)")
        buckets = sorted({r["bucket"] for r in rows} - {"NONE"})
        print(f"  {'Dimension':<10}" + "".join(f"{b:>9}" for b in buckets))
        for d in DIMENSIONS:
            cells = []
            for b in buckets:
                pos = [r for r in rows if r["bucket"] == b and r[f"gold_{d}"] == "True"]
                hit = sum(r[f"{d}_present"] == "True" for r in pos)
                cells.append(f"{hit}/{len(pos)}" if pos else "—")
            print(f"  {d:<10}" + "".join(f"{c:>9}" for c in cells))

        none = [r for r in rows if r["bucket"] == "NONE"]
        if none:
            print(f"\n  NONE bucket (control, {len(none)} rows), flagged present: "
                  + "  ".join(f"{d}={sum(r[f'{d}_present'] == 'True' for r in none)}" for d in DIMENSIONS))
        print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "score"):
        s = sub.add_parser(name)
        s.add_argument("--out-csv", type=Path, default=OUT_CSV)
    run = sub.choices["run"]
    run.add_argument("--condition", choices=[*CONDITIONS, "both"], default="both")
    run.add_argument("--include-none", action="store_true",
                     help="Also attack the 100 NONE-bucket rows (negatives for all three)")
    run.add_argument("--model", default="openai:gpt-5.5", help="Attacker model as provider:model")
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--concurrency", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "run":
        asyncio.run(_run(args))
    else:
        _score(args)


if __name__ == "__main__":
    main()
