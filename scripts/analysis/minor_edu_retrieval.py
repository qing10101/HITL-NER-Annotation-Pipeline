"""Screen a source JSONL for MINOR_EDU keyword candidates for human review.

Reads a source JSONL (schema: id, text, category, tier) and writes a ranked CSV
of every row that contains at least one positive MINOR_EDU trigger. Each output
row carries the matched triggers, detected education tier(s), anchor terms
(real-child markers that boost confidence), and exclusion terms (false-positive
signals that lower confidence), along with a net_score for sorting.

Scoring:
  +2  per unique trigger match
  +1  per unique anchor match  (possessives, kinship/age nouns, enrollment verbs)
  -1  per unique exclusion match  (occupations, higher-ed, animals, fiction,
                                   historical self-references, generic suitability)

High scores (≥ 3) are almost certainly genuine child-education mentions.
Low or negative scores are typically false positives (teacher reviews,
product-suitability language, "back in high school" self-references, etc.)
and can be dropped with --min-score.

Usage:
  # Full scan, sorted by confidence
  python scripts/minor_edu_retrieval.py data/sample_2000.jsonl

  # Only high-confidence rows
  python scripts/minor_edu_retrieval.py data/sample_2000.jsonl --min-score 3

  # Filter to a single education tier
  python scripts/minor_edu_retrieval.py data/sample_2000.jsonl --edu-tier high_school

  # Multiple tier filters, custom output path
  python scripts/minor_edu_retrieval.py data/sample_2000.jsonl \\
      --edu-tier elementary --edu-tier middle \\
      --out data/elem_mid_candidates.csv

  # Preserve original JSONL order instead of sorting by score
  python scripts/minor_edu_retrieval.py data/sample_2000.jsonl --no-sort
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

# ── tier labels ────────────────────────────────────────────────────────────── #

EARLY = "early_childhood"
ELEM  = "elementary"
MID   = "middle"
HIGH  = "high_school"
HOME  = "homeschool"

_TIER_ORDER = [EARLY, ELEM, MID, HIGH, HOME]   # priority for display

# ── lexicon ────────────────────────────────────────────────────────────────── #

# Positive triggers: (raw_pattern, tier)
TRIGGERS: list[tuple[str, str]] = [
    # ── early childhood ────────────────────────────────────────────────────
    (r"\bkindergarten(?:er|ner)?\b",                    EARLY),
    (r"\bkinder\b",                                     EARLY),
    (r"\bpre[-\s]?k\b",                                EARLY),
    (r"\bpre[-\s]?school(?:er)?\b",                    EARLY),
    (r"\bnursery\s+school\b",                           EARLY),

    # ── elementary — named tiers ───────────────────────────────────────────
    (r"\belementary\s+school(?:er)?\b",                ELEM),
    (r"\belementary\b",                                ELEM),
    (r"\bgrade\s+school\b",                            ELEM),
    (r"\bgrammar\s+school\b",                          ELEM),
    (r"\bprimary\s+school\b",                          ELEM),

    # ── elementary — grade numbers 1-5 ────────────────────────────────────
    (r"\b(?:1st|2nd|3rd|4th|5th"
     r"|first|second|third|fourth|fifth)\s+grade\b",   ELEM),
    (r"\bgrade\s+(?:[1-5]|one|two|three|four|five)\b", ELEM),
    (r"\b(?:1st|2nd|3rd|4th|5th)\s+grader\b",         ELEM),

    # ── middle / junior high — named tiers ────────────────────────────────
    (r"\bmiddle[-\s]?school(?:er)?\b",                 MID),
    (r"\bjunior\s+high\b",                             MID),
    (r"\bjr\.?\s*high\b",                             MID),

    # ── middle — grade numbers 6-8 ────────────────────────────────────────
    (r"\b(?:6th|7th|8th|sixth|seventh|eighth)\s+grade\b", MID),
    (r"\bgrade\s+(?:[6-8]|six|seven|eight)\b",             MID),
    (r"\b(?:6th|7th|8th)\s+grader\b",                     MID),

    # ── high school — named tiers ─────────────────────────────────────────
    (r"\bhigh[-\s]?school(?:er)?\b",                   HIGH),
    (r"\bhighschool\b",                                HIGH),

    # ── high school — grade numbers 9-12 ──────────────────────────────────
    (r"\b(?:9th|10th|11th|12th"
     r"|ninth|tenth|eleventh|twelfth)\s+grade\b",      HIGH),
    (r"\bgrade\s+(?:9|10|11|12|nine|ten|eleven|twelve)\b", HIGH),
    (r"\b(?:9th|10th|11th|12th)\s+grader\b",              HIGH),

    # ── high school — class-year nouns (ambiguous; see note in docstring) ──
    (r"\bfreshman\b",                                  HIGH),
    (r"\bsophomore\b",                                 HIGH),
    (r"\bjunior\b",                                    HIGH),
    (r"\bsenior\b",                                    HIGH),

    # ── homeschool ─────────────────────────────────────────────────────────
    (r"\bhome[-\s]?school(?:ing|ed|er)?\b",            HOME),
]

# Exclusion families — matches lower net_score
OCCUPATION = [
    r"\bteachers?\b",       r"\bprofessors?\b",  r"\bprincipal\b",
    r"\bcounselors?\b",     r"\bcoach(?:es)?\b", r"\bfaculty\b",
    r"\bstaff\b",           r"\badministrators?\b", r"\bsubstitutes?\b",
    r"\baides?\b",          r"\blibrarians?\b",
]
HIGHER_ED = [
    r"\bcollege\b",         r"\buniversity\b",   r"\bgrad\s+school\b",
    r"\bgraduate\b",        r"\btrade\s+school\b", r"\bvocational\b",
]
ANIMAL = [
    r"\bpuppy\s+school\b",
    r"\bdog\s+training(?:\s+school)?\b",
    r"\bobedience\s+school\b",
]
FICTION_MK = [
    r"\bcharacters?\b",     r"\bprotagonists?\b", r"\bnovels?\b",
    r"\bstory\b",           r"\bplot\b",           r"\bchapters?\b",
    r"\bauthors?\b",        r"\bset\s+in\b",
]
HISTORICAL = [
    r"\bwhen\s+i\s+was\b",  r"\bback\s+in\b",    r"\byears?\s+ago\b",
    r"\bgrowing\s+up\b",    r"\bi\s+was\s+in\b",
]
GENERIC = [
    r"\bgreat\s+for\b",     r"\bperfect\s+for\b", r"\bfor\s+any\b",
    r"\baccessible\s+to\b", r"\bsuitable\s+for\b", r"\bideal\s+for\b",
    r"\bages\s+\d+[-–]\d+\b",
]

# Anchor families — matches raise net_score
POSSESSIVE   = [r"\bmy\b", r"\bour\b"]
KIN_OR_AGE   = [
    r"\bsons?\b",           r"\bdaughters?\b",    r"\bkids?\b",
    r"\bchildren\b",        r"\bchild\b",          r"\bgrandsons?\b",
    r"\bgranddaughters?\b", r"\bgrandchild(?:ren)?\b", r"\bgrandkids?\b",
    r"\bnieces?\b",         r"\bnephews?\b",       r"\bcousins?\b",
    r"\btwins?\b",          r"\btoddlers?\b",
    r"\b\d+\s+years?\s+old\b", r"\b\d+yo\b",     r"\b\d+\s+months?\s+old\b",
]
ENROLL_VERB  = [
    r"\bis\s+in\b",         r"\bare\s+in\b",      r"\bstarting\b",
    r"\bstarted\b",         r"\benrolled\s+in\b",  r"\bgoes?\s+to\b",
    r"\bgoing\s+into\b",    r"\battends?\b",
]

# ── compile everything ─────────────────────────────────────────────────────── #

_FLAGS = re.IGNORECASE

_TRIGGER_RE: list[tuple[re.Pattern, str]] = [
    (re.compile(p, _FLAGS), tier) for p, tier in TRIGGERS
]

_EXCL_RE: dict[str, list[re.Pattern]] = {
    "occupation": [re.compile(p, _FLAGS) for p in OCCUPATION],
    "higher_ed":  [re.compile(p, _FLAGS) for p in HIGHER_ED],
    "animal":     [re.compile(p, _FLAGS) for p in ANIMAL],
    "fiction":    [re.compile(p, _FLAGS) for p in FICTION_MK],
    "historical": [re.compile(p, _FLAGS) for p in HISTORICAL],
    "generic":    [re.compile(p, _FLAGS) for p in GENERIC],
}

_ANCHOR_RE: dict[str, list[re.Pattern]] = {
    "possessive":  [re.compile(p, _FLAGS) for p in POSSESSIVE],
    "kin_or_age":  [re.compile(p, _FLAGS) for p in KIN_OR_AGE],
    "enroll_verb": [re.compile(p, _FLAGS) for p in ENROLL_VERB],
}

# ── screening logic ────────────────────────────────────────────────────────── #

def _unique_matches(text: str, family_map: dict[str, list[re.Pattern]]) -> list[str]:
    """Return deduplicated list of matched substrings (original casing) across
    all pattern families."""
    seen: set[str] = set()
    found: list[str] = []
    for patterns in family_map.values():
        for pat in patterns:
            for m in pat.finditer(text):
                key = m.group().lower()
                if key not in seen:
                    seen.add(key)
                    found.append(m.group())
    return found


def screen(text: str) -> dict | None:
    """Return a result dict if *text* contains at least one trigger, else None.

    Keys: triggers (list), edu_tiers (list), anchors (list),
          exclusions (list), net_score (int).
    """
    trigger_hits: list[str] = []
    tier_hits: set[str] = set()
    seen_keys: set[str] = set()

    for pat, tier in _TRIGGER_RE:
        for m in pat.finditer(text):
            key = m.group().lower()
            if key not in seen_keys:
                seen_keys.add(key)
                trigger_hits.append(m.group())
                tier_hits.add(tier)

    if not trigger_hits:
        return None

    anchor_hits    = _unique_matches(text, _ANCHOR_RE)
    excl_hits      = _unique_matches(text, _EXCL_RE)

    edu_tiers = [t for t in _TIER_ORDER if t in tier_hits]
    net_score = len(trigger_hits) * 2 + len(anchor_hits) - len(excl_hits)

    return {
        "triggers":   trigger_hits,
        "edu_tiers":  edu_tiers,
        "anchors":    anchor_hits,
        "exclusions": excl_hits,
        "net_score":  net_score,
    }


# ── output ─────────────────────────────────────────────────────────────────── #

_HEADER = [
    "row_num",
    "row_id",
    "category",
    "source_tier",
    "net_score",
    "edu_tiers",
    "triggers",
    "anchors",
    "exclusions",
    "text",
    "annotated_text",    # blank — human fills in
    "reviewer_notes",    # blank — optional human comments
]


# ── main ───────────────────────────────────────────────────────────────────── #

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Screen a source JSONL for MINOR_EDU keyword candidates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("jsonl", type=Path, help="source JSONL file to screen")
    ap.add_argument(
        "--out", type=Path, default=None,
        help="output CSV path (default: <input>_minor_edu.csv)",
    )
    ap.add_argument(
        "--min-score", type=int, default=-999,
        help="drop rows with net_score below this threshold (default: keep all)",
    )
    ap.add_argument(
        "--edu-tier", action="append", dest="edu_tiers",
        metavar="TIER",
        choices=[EARLY, ELEM, MID, HIGH, HOME],
        help=(
            "keep only rows matching this tier; "
            "repeat to allow multiple tiers "
            "(choices: early_childhood, elementary, middle, high_school, homeschool)"
        ),
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="max rows to write to the output CSV",
    )
    ap.add_argument(
        "--no-sort", action="store_true",
        help="preserve original JSONL order instead of sorting by net_score",
    )
    args = ap.parse_args()

    if not args.jsonl.exists():
        print(f"error: file not found: {args.jsonl}", file=sys.stderr)
        return 1

    out_path = args.out or args.jsonl.with_name(args.jsonl.stem + "_minor_edu.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── scan ──────────────────────────────────────────────────────────────── #

    candidates: list[dict] = []
    total_scanned = 0
    skipped_malformed = 0

    with args.jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped_malformed += 1
                continue

            total_scanned += 1
            text = obj.get("text", "")
            result = screen(text)
            if result is None:
                continue

            # tier filter
            if args.edu_tiers and not any(
                t in result["edu_tiers"] for t in args.edu_tiers
            ):
                continue

            # score filter
            if result["net_score"] < args.min_score:
                continue

            candidates.append({
                "id":           obj.get("id", ""),
                "category":     obj.get("category", ""),
                "source_tier":  obj.get("tier", ""),
                "text":         text,
                **result,
            })

    # ── sort & cap ────────────────────────────────────────────────────────── #

    if not args.no_sort:
        candidates.sort(key=lambda r: r["net_score"], reverse=True)

    if args.limit:
        candidates = candidates[: args.limit]

    # ── write CSV ─────────────────────────────────────────────────────────── #

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_HEADER)
        for i, row in enumerate(candidates, 1):
            writer.writerow([
                i,
                row["id"],
                row["category"],
                row["source_tier"],
                row["net_score"],
                ", ".join(row["edu_tiers"]),
                ", ".join(row["triggers"]),
                ", ".join(row["anchors"]),
                ", ".join(row["exclusions"]),
                row["text"],
                "",   # annotated_text
                "",   # reviewer_notes
            ])

    # ── summary ───────────────────────────────────────────────────────────── #

    tier_counts: Counter[str] = Counter()
    for row in candidates:
        for t in row["edu_tiers"]:
            tier_counts[t] += 1

    print(f"scanned  : {total_scanned:>6} rows")
    if skipped_malformed:
        print(f"malformed: {skipped_malformed:>6} lines skipped", file=sys.stderr)
    print(f"matched  : {len(candidates):>6} rows -> {out_path}")
    print()
    print(f"  {'tier':<20}  {'rows':>5}")
    print(f"  {'-'*20}  {'-'*5}")
    for tier in _TIER_ORDER:
        if tier_counts[tier]:
            print(f"  {tier:<20}  {tier_counts[tier]:>5}")

    if candidates:
        scores = [r["net_score"] for r in candidates]
        print()
        print(f"  net_score range : {min(scores)} – {max(scores)}")
        high = sum(1 for s in scores if s >= 3)
        med  = sum(1 for s in scores if 0 <= s < 3)
        low  = sum(1 for s in scores if s < 0)
        print(f"  high (≥ 3)      : {high}")
        print(f"  medium (0–2)    : {med}")
        print(f"  low (< 0)       : {low}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
