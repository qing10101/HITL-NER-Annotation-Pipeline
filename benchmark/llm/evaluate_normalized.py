"""
evaluate_normalized.py

Normalization-tolerant variant of evaluate.py. Instead of trusting the
predicted_entities_json column (which annotate.py produced with the pipeline's
strict tag parser, pipeline/parser.py), this script re-parses the raw
predicted_tagged_text column with a customised parser that normalizes the review
text before counting offsets:

    1. every literal chunk is converted to plain ASCII (curly quotes -> straight
       quotes, en/em dashes -> "-", ellipsis -> "...", accented letters folded via
       NFKD, any remaining non-ASCII dropped), and
    2. ALL whitespace is stripped (spaces, tabs, newlines, non-breaking spaces),

so offsets are counted in a space-free, ASCII-only coordinate system. Gold spans
are remapped into the same coordinate system from the gold CSV's raw_text, which
keeps the two sides comparable.

Why: evaluate.py scores exact (label, start, end) matches in raw_text
coordinates. When a model rewrites formatting -- normalizes a curly quote,
collapses "  " to " ", drops a "<br />" newline -- every offset after the first
edit shifts, and the row is double-penalized (false positives AND false
negatives) even though the labeled spans are substantively right (see
check_parse_errors.py's "mutated" drift class). Scoring in normalized
coordinates forgives exactly that class of drift: whitespace and Unicode/ASCII
formatting differences. Real text mutations (paraphrases, dropped words) still
misalign and still score as errors.

Rows marked FAILED (all annotate.py retries exhausted) are scored as recall
misses, exactly as in evaluate.py. Rows whose tag structure is malformed
(unbalanced/mismatched tags) parse to zero entities, matching annotate.py's
behaviour, and their count is reported per condition.

Usage:
    # Score one prediction set
    python evaluate_normalized.py --gold output/gold_standard_merged.csv \
        --pred predictions.csv

    # Compare zero-shot vs retriever-equipped conditions
    python evaluate_normalized.py --gold output/gold_standard_merged.csv \
        --pred zero_shot=predictions_k0.csv \
        --pred retriever=predictions_k8.csv

    # Predictions made with a 3-dimension guideline: collapse gold's 5 labels
    python evaluate_normalized.py --gold output/gold_standard_merged.csv \
        --collapse-3dim --pred retriever=predictions_k8.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

# evaluate.py is a sibling; reuse its scoring/reporting so the two scripts can
# only ever disagree about coordinates, never about the metric.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (COARSE_3DIM, FAILED_MARKER, THREE_DIM_LABELS,  # noqa: E402
                      parse_pred_arg, print_report, score)


class TagParseError(ValueError):
    """Raised when tag structure is malformed (unbalanced / mismatched)."""


# ---------------------------------------------------------------------------
# Normalization: ASCII-fold + strip all whitespace
# ---------------------------------------------------------------------------

# Typographic characters NFKD does not decompose to ASCII; folded explicitly so
# a model that "fixes" punctuation still lands on the same normalized text.
# Written as \uXXXX escapes: several of these are invisible or indistinguishable
# from their ASCII counterpart in an editor.
_PUNCT_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",   # curly single quotes
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',   # curly double quotes
    "\u2032": "'", "\u2033": '"',                                 # prime marks
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",   # hyphens / en dash
    "\u2014": "-", "\u2015": "-", "\u2212": "-",                  # em dash / bar / minus
    "\u2026": "...",                                              # ellipsis
    "\u00a0": " ", "\u2007": " ", "\u202f": " ",                  # hard spaces
    "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",       # zero-width chars
})


def to_ascii(text: str) -> str:
    """Fold text to plain ASCII: map typographic punctuation, decompose accents
    (NFKD), and drop whatever still isn't ASCII."""
    text = text.translate(_PUNCT_MAP)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def normalize_text(text: str) -> str:
    """ASCII-fold, then strip ALL whitespace. This is the coordinate system every
    span in this script is expressed in."""
    return re.sub(r"\s+", "", to_ascii(text))


# ---------------------------------------------------------------------------
# Customised parser: same left-to-right sweep as pipeline/parser.py, but each
# literal chunk is normalized before it contributes to the offset count.
# ---------------------------------------------------------------------------

def parse_tagged_text_normalized(tagged_text: str, tagset) -> tuple[str, list[dict]]:
    """Parse inline-tagged text into (normalized_clean_text, spans).

    Offsets are 0-based, half-open, and counted in normalized (ASCII-folded,
    whitespace-free) coordinates: normalized_clean_text[start:end] == span text.
    Raises TagParseError on unbalanced or mismatched tags.
    """
    tag_re = re.compile(r"<(/?)(" + "|".join(tagset) + r")>")
    clean_parts: list[str] = []
    clean_len = 0
    spans: list[dict] = []
    stack: list[tuple[str, int]] = []  # (label, start_offset)
    cursor = 0

    for m in tag_re.finditer(tagged_text):
        chunk = normalize_text(tagged_text[cursor:m.start()])
        if chunk:
            clean_parts.append(chunk)
            clean_len += len(chunk)
        cursor = m.end()

        is_close = m.group(1) == "/"
        label = m.group(2)
        if not is_close:
            stack.append((label, clean_len))
        else:
            if not stack:
                raise TagParseError(f"Closing </{label}> with no matching open tag")
            open_label, start = stack.pop()
            if open_label != label:
                raise TagParseError(f"Tag mismatch: <{open_label}> closed by </{label}>")
            spans.append({"label": label, "text": "", "start": start, "end": clean_len})

    trailing = normalize_text(tagged_text[cursor:])
    if trailing:
        clean_parts.append(trailing)

    if stack:
        unclosed = ", ".join(f"<{lbl}>" for lbl, _ in stack)
        raise TagParseError(f"Unclosed tag(s): {unclosed}")

    clean_text = "".join(clean_parts)
    for span in spans:
        span["text"] = clean_text[span["start"]:span["end"]]
    spans.sort(key=lambda s: (s["start"], s["end"]))
    return clean_text, spans


def remap_gold_entities(raw_text: str, entities: list[dict]) -> list[dict]:
    """Re-express gold spans (offsets into raw_text) in normalized coordinates.

    norm_start is the normalized length of everything before the span; the span's
    own normalized text supplies its width. Exact by construction, since
    normalize_text is per-character (each input char maps to 0+ output chars).
    """
    remapped = []
    for e in entities:
        norm_start = len(normalize_text(raw_text[:e["start"]]))
        norm_text = normalize_text(raw_text[e["start"]:e["end"]])
        if not norm_text:  # span was pure whitespace/non-ASCII; nothing left to match
            continue
        remapped.append({"label": e["label"], "text": norm_text,
                         "start": norm_start, "end": norm_start + len(norm_text)})
    return remapped


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_gold_normalized(path: str) -> dict[str, list[dict]]:
    """Gold CSV (row_id, raw_text, entities_json) -> normalized-coordinate spans."""
    gold: dict[str, list[dict]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entities = json.loads(row["entities_json"] or "[]")
            gold[row["row_id"]] = remap_gold_entities(row["raw_text"] or "", entities)
    return gold


def load_pred_normalized(path: str, tagset) -> tuple[dict[str, list[dict]], set[str], set[str], dict[str, int]]:
    """Re-parse predicted_tagged_text with the normalizing parser.

    Returns (entities_by_id, failed_ids, parse_error_ids, diagnostics).
    FAILED rows (annotate.py exhausted retries) and rows with malformed tag
    structure both map to [] so their gold spans score as false negatives;
    diagnostics counts rows whose normalized stripped text exactly matches the
    normalized raw_text (i.e. rows this scoring rescues from formatting drift).
    """
    pred: dict[str, list[dict]] = {}
    failed: set[str] = set()
    parse_errors: set[str] = set()
    norm_clean = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = row["row_id"]
            if row["predicted_entities_json"] == FAILED_MARKER:
                pred[rid] = []
                failed.add(rid)
                continue
            tagged = row["predicted_tagged_text"] or ""
            try:
                clean_text, spans = parse_tagged_text_normalized(tagged, tagset)
            except TagParseError:
                pred[rid] = []
                parse_errors.add(rid)
                continue
            # Zero-width spans (tagged text that normalized away) can't match anything.
            pred[rid] = [s for s in spans if s["end"] > s["start"]]
            if clean_text == normalize_text(row.get("raw_text") or ""):
                norm_clean += 1
    return pred, failed, parse_errors, {"norm_clean": norm_clean}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score annotate.py predictions against gold in normalized "
                    "(ASCII-folded, whitespace-free) coordinates, re-parsing "
                    "predicted_tagged_text instead of using the pipeline parser's "
                    "stored offsets."
    )
    ap.add_argument("--gold", required=True,
                    help="Gold-standard CSV (row_id, raw_text, entities_json, ...). "
                         "raw_text is required to remap gold offsets.")
    ap.add_argument("--pred", action="append", required=True,
                    help="Predictions CSV (row_id, raw_text, predicted_tagged_text, ...), "
                         "optionally 'name=path.csv' to label it in the report. "
                         "Repeatable to compare conditions.")
    ap.add_argument("--labels", nargs="+", default=THREE_DIM_LABELS,
                    help=f"Tagset to parse predictions against (default: "
                         f"{' '.join(THREE_DIM_LABELS)}). Must match the label scheme "
                         "the guideline told the model to emit.")
    ap.add_argument("--collapse-3dim", action="store_true",
                    help="Collapse the gold corpus's 5 fine labels onto the 3 coarse "
                         "dimensions (MINOR, GENDER, KINSHIP) before scoring.")
    args = ap.parse_args()

    gold = load_gold_normalized(args.gold)
    gold_map = COARSE_3DIM if args.collapse_3dim else None
    if args.collapse_3dim:
        print("Scoring on 3 collapsed dimensions: " + ", ".join(THREE_DIM_LABELS))
    print("Coordinates: ASCII-folded, all whitespace stripped "
          "(re-parsed from predicted_tagged_text).")

    results = {}
    for raw in args.pred:
        name, path = parse_pred_arg(raw)
        pred, failed_ids, parse_error_ids, diag = load_pred_normalized(path, args.labels)
        results[name] = score(gold, pred, failed_ids, gold_label_map=gold_map)
        print_report(name, results[name])
        parseable = len(pred) - len(failed_ids)
        print(f"  re-parse: {len(parse_error_ids)} malformed-tag row(s) scored as empty; "
              f"{diag['norm_clean']}/{parseable} parseable row(s) match raw_text exactly "
              f"after normalization")

    if len(results) > 1:
        print("\n=== COMPARISON (micro-avg, normalized coordinates) ===")
        width = max(len(name) for name in results)
        for name, result in results.items():
            m = result["micro"]
            failed = result.get("failed_rows", 0)
            fail_note = f"  [{failed} failed row(s)]" if failed else ""
            print(f"  {name:<{width}}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
                  f"F1={m['f1']:.3f}{fail_note}")


if __name__ == "__main__":
    main()
