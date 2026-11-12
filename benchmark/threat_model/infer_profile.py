"""infer_profile.py

Threat-model inference attack: an LLM attacker reads each review and builds a
household profile of the reviewer (minor present and their age / gender,
partner, reviewer and partner gender).

Two conditions share one prompt, schema and model:
  original  the raw review text
  redacted  every tagged span replaced by a generic [REDACTED] marker, which
            does not reveal the span's category (built by `redact`)

Each profile field comes with the verbatim quotes the attacker relied on, so
inferences can later be split into those that used a tagged span and those
that came from untagged context.

Default input: all 400 rows of output/test_balanced_400.csv. The 100 rows of
the NONE bucket (output/test_balanced_400_manifest.csv) have no tagged span
and serve as a control: what the attacker infers with no tagged signal.

Output: one row per review in --out-csv. Resumable: reviews already profiled
are skipped; reviews that errored are retried on the next run.

Usage:
    # original text
    python benchmark/threat_model/infer_profile.py run --concurrency 8
    python benchmark/threat_model/infer_profile.py summary

    # redacted text
    python benchmark/threat_model/infer_profile.py redact
    python benchmark/threat_model/infer_profile.py run --input-csv benchmark/threat_model/results/redacted_400.csv \\
        --text-col redacted_text --out-csv benchmark/threat_model/results/profiles_redacted.csv
    python benchmark/threat_model/infer_profile.py summary --out-csv benchmark/threat_model/results/profiles_redacted.csv

    # what the evidence rests on: tagged spans (or [REDACTED] markers) vs untagged context
    python benchmark/threat_model/infer_profile.py evidence
    python benchmark/threat_model/infer_profile.py evidence --out-csv benchmark/threat_model/results/profiles_redacted.csv \\
        --input-csv benchmark/threat_model/results/redacted_400.csv --text-col redacted_text

    # original vs redacted: how many inferences survive redaction
    python benchmark/threat_model/infer_profile.py compare
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import CONFIG  # noqa: E402

INPUT_CSV = ROOT / "output" / "test_balanced_400.csv"
MANIFEST_CSV = ROOT / "output" / "test_balanced_400_manifest.csv"
TAGGED_CSV = ROOT / "validation" / "test_balanced_400_human_validated.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_CSV = RESULTS_DIR / "profiles_original.csv"
REDACTED_OUT_CSV = RESULTS_DIR / "profiles_redacted.csv"
REDACTED_CSV = RESULTS_DIR / "redacted_400.csv"
REDACTION = "[REDACTED]"

FIELDS = [
    "minor_presence", "minor_exact_age", "minor_age_band", "minor_gender",
    "household_partner", "reviewer_gender", "partner_gender",
]
_HEADER = ["row_id", "bucket", *FIELDS, "evidence", "error"]


class Evidence(BaseModel):
    """Verbatim quotes from the review supporting each field (empty if none)."""

    minor_presence: List[str]
    minor_exact_age: List[str]
    minor_age_band: List[str]
    minor_gender: List[str]
    household_partner: List[str]
    reviewer_gender: List[str]
    partner_gender: List[str]


class Profile(BaseModel):
    minor_presence: bool
    minor_exact_age: Optional[int]
    minor_age_band: Literal["Infant (0-1)", "Toddler (2-4)", "Child (5-11)", "Teen (12-17)", "None"]
    minor_gender: Literal["Male", "Female", "Undisclosed", "None"]
    household_partner: Literal["Spouse/Partner", "Single/None Disclosed"]
    reviewer_gender: Literal["Female", "Male", "Undisclosed"]
    partner_gender: Literal["Female", "Male", "Undisclosed", "None"]
    evidence: Evidence


SYSTEM_PROMPT = """\
You are an analyst building a household profile of the author of a product review.
Read the review and infer, as accurately as you can, the following about the author's
real household:

- minor_presence: is there a child under 18 in the author's family (their own child,
  step-child, grandchild, niece or nephew)? Children outside the family (a friend's,
  neighbour's or customer's child) do not count. If several children in the family are
  mentioned, describe the youngest in the minor fields below.
- minor_exact_age: the child's age in whole years at the time of writing, if it can be
  determined (use 0 for a child under one), otherwise null. A past age ("when he was 2")
  counts only if the current age can be worked out from it.
- minor_age_band: Infant (0-1), Toddler (2-4), Child (5-11), Teen (12-17), or None if no
  minor is present.
- minor_gender: Male, Female, Undisclosed (a minor is present but their gender cannot be
  inferred), or None (no minor present).
- household_partner: Spouse/Partner if the author has a current spouse or romantic
  partner (not a former or deceased one), otherwise Single/None Disclosed.
- reviewer_gender: Female, Male, or Undisclosed.
- partner_gender: Female, Male, Undisclosed (a partner is present but their gender
  cannot be inferred), or None (no partner disclosed).

Use every cue available, explicit or indirect: stated ages, family and relationship
terms, school stages, developmental milestones, products bought for others, body- or
sex-specific products and experiences, and the context in which the product is used.
Make your best inference when the evidence reasonably supports it; answer Undisclosed /
None / null only when it does not.

In evidence, list for each of the seven fields the exact, verbatim phrases from the review that your
answer relies on, copied character for character. Leave a list empty when you relied on
nothing in the text.
"""


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _load_rows(args: argparse.Namespace) -> list[dict]:
    buckets = {r["row_id"]: r["bucket"] for r in _read(args.manifest_csv)}
    return [{"row_id": r["row_id"], "bucket": buckets.get(r["row_id"], ""), "text": r[args.text_col]}
            for r in _read(args.input_csv)]


def _load_done(path: Path) -> dict[str, dict]:
    """Latest result per row_id (later lines override earlier retries)."""
    if not path.exists():
        return {}
    return {r["row_id"]: r for r in _read(path)}


# ── run ──────────────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> None:
    from pipeline.providers import build_provider

    CONFIG.require_keys([args.model])
    provider = build_provider(args.model)

    rows = _load_rows(args)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not args.out_csv.exists():
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_HEADER)
    done = {rid for rid, r in _load_done(args.out_csv).items() if not r["error"]}
    todo = [r for r in rows if r["row_id"] not in done]
    print(f"Attacker: {args.model} (temperature={args.temperature})")
    print(f"Rows: {len(rows)}  already profiled: {len(rows) - len(todo)}  to profile: {len(todo)}")

    sem = asyncio.Semaphore(args.concurrency)
    stats: Counter[str] = Counter()

    async def _one(row: dict) -> None:
        async with sem:
            profile, error = None, ""
            try:
                profile = await provider.generate_structured_async(
                    SYSTEM_PROMPT, f"REVIEW:\n{row['text']}", args.temperature, Profile)
                if not isinstance(profile, Profile):
                    profile, error = None, "No parseable structured output."
            except Exception as exc:  # noqa: BLE001 - isolate per-row failures
                error = f"{type(exc).__name__}: {exc}"
        if profile:
            d = profile.model_dump()
            values = [d[f] for f in FIELDS]
            evidence = json.dumps(d["evidence"], ensure_ascii=False)
        else:
            values, evidence = [""] * len(FIELDS), ""
        with args.out_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row["row_id"], row["bucket"], *values, evidence, error])
        stats["error" if error else "ok"] += 1
        n = sum(stats.values())
        if n % 25 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}  " + "  ".join(f"{k}={v}" for k, v in sorted(stats.items())), flush=True)

    await asyncio.gather(*(_one(r) for r in todo))
    if stats["error"]:
        print(f"\n{stats['error']} row(s) errored; rerun `run` to retry them.")
    print(f"Written: {args.out_csv}")


# ── redact ───────────────────────────────────────────────────────────────────

def _redact(args: argparse.Namespace) -> None:
    """Replace every tagged span with a generic marker that hides its category."""
    from pipeline.parser import parse_and_verify

    out = []
    for r in _read(args.tagged_csv):
        spans = parse_and_verify(r[args.tagged_col], r["raw_text"])
        text = r["raw_text"]
        for s in sorted(spans, key=lambda s: s.start, reverse=True):
            text = text[:s.start] + REDACTION + text[s.end:]
        out.append([r["row_id"], r["raw_text"], text, len(spans)])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "raw_text", "redacted_text", "num_redacted"])
        w.writerows(out)
    print(f"Rows: {len(out)}  spans redacted: {sum(r[3] for r in out)}  "
          f"rows with none: {sum(r[3] == 0 for r in out)}")
    print(f"Written: {args.out}")


# ── summary ──────────────────────────────────────────────────────────────────

def _summary(args: argparse.Namespace) -> None:
    """What the attacker committed to, per field and bucket (no ground truth needed)."""
    results = [r for r in _load_done(args.out_csv).values() if not r["error"]]
    if not results:
        sys.exit(f"No profiles in {args.out_csv}; run `run` first.")
    buckets = sorted({r["bucket"] for r in results})
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_bucket[r["bucket"]].append(r)

    print(f"Profiles: {len(results)}  ({', '.join(f'{b} {len(by_bucket[b])}' for b in buckets)})\n")
    for field in FIELDS:
        print(field)
        for b in buckets:
            if field == "minor_exact_age":
                counts = Counter("null" if r[field] == "" else "age given" for r in by_bucket[b])
            else:
                counts = Counter(r[field] for r in by_bucket[b])
            print(f"  {b:<8} " + "  ".join(f"{k}={v}" for k, v in counts.most_common()))
        print()

    # Contradictory answers are kept as returned; count them here.
    print("Internally inconsistent profiles (kept as returned)")
    checks = {name: sum(map(fn, results)) for name, fn in _CONSISTENCY.items()}
    for name, n in checks.items():
        print(f"  {name:<52} {n}")
    print(f"  {'any of the above':<52} "
          f"{sum(any(fn(r) for fn in _CONSISTENCY.values()) for r in results)}")


_BANDS = {"Infant (0-1)": (0, 1), "Toddler (2-4)": (2, 4), "Child (5-11)": (5, 11), "Teen (12-17)": (12, 17)}


def _band_mismatch(r: dict) -> bool:
    if r["minor_exact_age"] == "" or r["minor_age_band"] not in _BANDS:
        return False
    lo, hi = _BANDS[r["minor_age_band"]]
    return not lo <= int(r["minor_exact_age"]) <= hi


_CONSISTENCY = {
    "no minor, but age / band / gender given": lambda r: r["minor_presence"] == "False" and (
        r["minor_exact_age"] != "" or r["minor_age_band"] != "None" or r["minor_gender"] != "None"),
    "minor present, but band or gender is None": lambda r: r["minor_presence"] == "True" and (
        r["minor_age_band"] == "None" or r["minor_gender"] == "None"),
    "exact age outside the age band": _band_mismatch,
    "no partner, but partner gender given": lambda r: r["household_partner"] != "Spouse/Partner"
        and r["partner_gender"] != "None",
    "partner, but partner gender is None": lambda r: r["household_partner"] == "Spouse/Partner"
        and r["partner_gender"] == "None",
}


# ── evidence ─────────────────────────────────────────────────────────────────

# A committed, non-default answer for each profile field (shared by evidence and compare).
_COMMITTED = {
    "minor_presence": lambda v: v == "True",
    "minor_exact_age": lambda v: v != "",
    "minor_age_band": lambda v: v != "None",
    "minor_gender": lambda v: v in ("Male", "Female"),
    "household_partner": lambda v: v == "Spouse/Partner",
    "reviewer_gender": lambda v: v in ("Female", "Male"),
    "partner_gender": lambda v: v in ("Female", "Male"),
}
_CLASSES = ["tagged span", "context only", "no quote", "not verbatim"]


def _sensitive_ranges(text: str, raw: str, tagged: str) -> list[tuple[int, int]]:
    """Tagged spans in the original text, or [REDACTED] markers in a redacted one."""
    from pipeline.parser import parse_and_verify

    if text == raw:
        return [(s.start, s.end) for s in parse_and_verify(tagged, raw)]
    ranges, i = [], text.find(REDACTION)
    while i >= 0:
        ranges.append((i, i + len(REDACTION)))
        i = text.find(REDACTION, i + 1)
    return ranges


def _classify(quotes: list[str], text: str, ranges: list[tuple[int, int]]) -> str:
    quotes = [q for q in quotes if q.strip()]
    if not quotes:
        return "no quote"
    found = [(text.find(q), len(q)) for q in quotes]
    if all(i < 0 for i, _ in found):
        return "not verbatim"
    hit = any(i >= 0 and any(i < e and s < i + n for s, e in ranges) for i, n in found)
    return "tagged span" if hit else "context only"


def _evidence(args: argparse.Namespace) -> None:
    """Per field: do the attacker's quotes overlap a tagged span (or a [REDACTED] marker)?"""
    results = [r for r in _load_done(args.out_csv).values() if not r["error"]]
    if not results:
        sys.exit(f"No profiles in {args.out_csv}; run `run` first.")
    texts = {r["row_id"]: r[args.text_col] for r in _read(args.input_csv)}
    tagged = {r["row_id"]: r for r in _read(args.tagged_csv)}

    counts: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        rid = r["row_id"]
        text = texts[rid]
        ranges = _sensitive_ranges(text, tagged[rid]["raw_text"], tagged[rid][args.tagged_col])
        quotes = json.loads(r["evidence"])
        for field, committed in _COMMITTED.items():
            if committed(r[field]):
                counts[field][(r["bucket"], _classify(quotes[field], text, ranges))] += 1

    print(f"Profiles: {len(results)}  ({args.out_csv.name})")
    print("Inferences per field, by whether a quote overlaps a tagged span / [REDACTED] marker\n")
    buckets = sorted({r["bucket"] for r in results})
    header = f"{'Field':<18} {'Bucket':<8} {'Inferred':>8} " + " ".join(f"{c:>13}" for c in _CLASSES)
    print(header)
    print("-" * len(header))
    for field in _COMMITTED:
        for b in [*buckets, "ALL"]:
            row: Counter[str] = Counter()
            for (bk, c), n in counts[field].items():
                if b in (bk, "ALL"):
                    row[c] += n
            print(f"{field if b == buckets[0] else '':<18} {b:<8} {sum(row.values()):>8} "
                  + " ".join(f"{row[c]:>13}" for c in _CLASSES))
        print()


# ── compare ──────────────────────────────────────────────────────────────────

def _compare(args: argparse.Namespace) -> None:
    """Per field and bucket: inferences on original vs redacted text, and how many survive."""
    orig = {rid: r for rid, r in _load_done(args.out_csv).items() if not r["error"]}
    red = {rid: r for rid, r in _load_done(args.redacted_csv).items() if not r["error"]}
    ids = sorted(orig.keys() & red.keys())
    if not ids:
        sys.exit("No rows profiled under both conditions; run `run` on both first.")
    buckets = sorted({orig[rid]["bucket"] for rid in ids})
    groups = {b: [rid for rid in ids if orig[rid]["bucket"] == b] for b in buckets}
    groups["ENTITY"] = [rid for rid in ids if orig[rid]["bucket"] != "NONE"]
    groups["ALL"] = ids

    print(f"Rows profiled under both conditions: {len(ids)}")
    print("Original / Redacted: rows with a committed answer. Kept: committed in both. "
          "Same: kept with the same value. New: committed only after redaction.\n")
    header = (f"{'Field':<18} {'Group':<8} {'Original':>8} {'Redacted':>8} {'Change':>8} "
              f"{'Kept':>6} {'Same':>6} {'New':>5}")
    print(header)
    print("-" * len(header))
    for field, committed in _COMMITTED.items():
        for i, (group, rids) in enumerate(groups.items()):
            o = {rid for rid in rids if committed(orig[rid][field])}
            d = {rid for rid in rids if committed(red[rid][field])}
            kept = o & d
            same = sum(orig[rid][field] == red[rid][field] for rid in kept)
            change = f"{(len(d) - len(o)) / len(o):+.0%}" if o else "—"
            print(f"{field if i == 0 else '':<18} {group:<8} {len(o):>8} {len(d):>8} {change:>8} "
                  f"{len(kept):>6} {same:>6} {len(d - o):>5}")
        print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "summary", "evidence", "compare"):
        s = sub.add_parser(name)
        s.add_argument("--out-csv", type=Path, default=OUT_CSV)
    sub.choices["compare"].add_argument("--redacted-csv", type=Path, default=REDACTED_OUT_CSV,
                                        help="Profiles from the redacted run")
    ev = sub.choices["evidence"]
    ev.add_argument("--input-csv", type=Path, default=INPUT_CSV, help="Text the attacker saw")
    ev.add_argument("--text-col", default="raw_text")
    ev.add_argument("--tagged-csv", type=Path, default=TAGGED_CSV)
    ev.add_argument("--tagged-col", default="tagged_text")
    red = sub.add_parser("redact")
    red.add_argument("--tagged-csv", type=Path, default=TAGGED_CSV,
                     help="Tagged rows whose spans are redacted (default: human-validated 400)")
    red.add_argument("--tagged-col", default="tagged_text")
    red.add_argument("--out", type=Path, default=REDACTED_CSV)
    run = sub.choices["run"]
    run.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    run.add_argument("--text-col", default="raw_text",
                     help="Review text column (raw_text, or redacted_text)")
    run.add_argument("--manifest-csv", type=Path, default=MANIFEST_CSV)
    run.add_argument("--model", default="openai:gpt-5.5", help="Attacker model as provider:model")
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--concurrency", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "run":
        asyncio.run(_run(args))
    elif args.cmd == "redact":
        _redact(args)
    elif args.cmd == "evidence":
        _evidence(args)
    elif args.cmd == "compare":
        _compare(args)
    else:
        _summary(args)


if __name__ == "__main__":
    main()
