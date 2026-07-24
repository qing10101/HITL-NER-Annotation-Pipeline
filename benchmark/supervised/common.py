"""Shared utilities for the supervised NER trainers (benchmark/supervised).

Everything in this module is stdlib-only so that data preparation and the
unit tests run without torch / spacy / transformers installed. The heavy
imports live inside the individual train_*.py scripts.

Span convention matches the rest of the repo (benchmark/llm/evaluate.py):
an entity is {"label", "text", "start", "end"} with character offsets on
raw_text, end-exclusive.
"""
from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Canonical label set (most frequent first, per gold_standard_merged.csv).
LABELS = ["FAM_KIN", "MINOR_AGE", "GEN_NOUN", "GEN_PHYS", "MINOR_EDU"]

# Rarest-first order used to pick a stratum for each product group when
# splitting, so thin labels (MINOR_EDU: ~97, GEN_PHYS: ~164) are spread
# proportionally across train/dev/test.
RARITY_ORDER = ["MINOR_EDU", "GEN_PHYS", "GEN_NOUN", "MINOR_AGE", "FAM_KIN"]

# GLiNER matches spans against natural-language label names, so the label
# codes are exposed to it as short definitional phrases (from README.md).
# The exact same phrases must be used at train and predict time.
GLINER_LABEL_PHRASES = {
    "FAM_KIN": "kinship term for the reviewer's family member",
    "MINOR_AGE": "age or developmental stage of a child under 18",
    "GEN_NOUN": "gendered noun for the reviewer or their romantic partner",
    "GEN_PHYS": "sex-specific physiological condition or milestone",
    "MINOR_EDU": "educational tier exclusive to minors",
}
PHRASE_TO_LABEL = {v: k for k, v in GLINER_LABEL_PHRASES.items()}

# BIO tagset shared by the BiLSTM-CRF and SpanMarker trainers.
BIO_TAGS = ["O"] + [f"{p}-{lbl}" for lbl in LABELS for p in ("B", "I")]
BIO_TAG_TO_ID = {t: i for i, t in enumerate(BIO_TAGS)}

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


@dataclass
class Example:
    row_id: str
    text: str
    entities: list[dict]
    raw_row: dict  # original CSV row, carried through so splits keep the schema


# ---------------------------------------------------------------------------
# Loading / writing
# ---------------------------------------------------------------------------

def load_examples(csv_path: str | Path) -> list[Example]:
    """Load a gold-format CSV (row_id, raw_text, ..., entities_json)."""
    examples = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            examples.append(Example(
                row_id=row["row_id"],
                text=row["raw_text"],
                entities=json.loads(row["entities_json"] or "[]"),
                raw_row=dict(row),
            ))
    return examples


def write_examples_csv(path: str | Path, examples: list[Example]) -> None:
    """Write examples back out with the same columns as the source CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not examples:
        raise ValueError(f"refusing to write empty split: {path}")
    fieldnames = list(examples[0].raw_row.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ex in examples:
            writer.writerow(ex.raw_row)


def write_predictions_csv(path: str | Path, rows: list[tuple[str, list[dict]]]) -> None:
    """Write (row_id, entities) pairs in the format evaluate.py expects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "predicted_entities_json"])
        for row_id, entities in rows:
            writer.writerow([row_id, json.dumps(entities, ensure_ascii=False)])


# ---------------------------------------------------------------------------
# Tokenization and BIO conversion
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[tuple[str, int, int]]:
    """Split into (token, char_start, char_end) triples.

    Words and single punctuation marks; offsets index into the original text,
    so spans survive the round trip to tags and back exactly.
    """
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def spans_to_bio(
    tokens: list[tuple[str, int, int]],
    entities: list[dict],
) -> tuple[list[str], int, int]:
    """Project char-offset entities onto BIO tags over `tokens`.

    Returns (tags, n_adjusted, n_dropped). A span whose boundaries fall
    inside a token is expanded to full token boundaries (counted in
    n_adjusted); a span matching no token, or overlapping an already-tagged
    one, is dropped (counted in n_dropped).
    """
    tags = ["O"] * len(tokens)
    adjusted = dropped = 0
    for ent in sorted(entities, key=lambda e: (e["start"], e["end"])):
        idxs = [
            i for i, (_, s, e) in enumerate(tokens)
            if s < ent["end"] and e > ent["start"]
        ]
        if not idxs or any(tags[i] != "O" for i in idxs):
            dropped += 1
            continue
        if tokens[idxs[0]][1] != ent["start"] or tokens[idxs[-1]][2] != ent["end"]:
            adjusted += 1
        tags[idxs[0]] = f"B-{ent['label']}"
        for i in idxs[1:]:
            tags[i] = f"I-{ent['label']}"
    return tags, adjusted, dropped


def bio_to_spans(
    text: str,
    tokens: list[tuple[str, int, int]],
    tags: list[str],
) -> list[dict]:
    """Rebuild char-offset entities from BIO tags. Tolerates I- without B-."""
    spans = []
    i = 0
    while i < len(tags):
        if tags[i] == "O":
            i += 1
            continue
        label = tags[i][2:]
        first = i
        i += 1
        while i < len(tags) and tags[i] == f"I-{label}":
            i += 1
        start, end = tokens[first][1], tokens[i - 1][2]
        spans.append({"label": label, "text": text[start:end], "start": start, "end": end})
    return spans


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def group_key(row_id: str) -> str:
    """Product id prefix of a row_id like 'B0B93WDZ6J_64196'."""
    return row_id.split("_", 1)[0]


def _stratum(examples: list[Example]) -> str:
    present = {e["label"] for ex in examples for e in ex.entities}
    for lbl in RARITY_ORDER:
        if lbl in present:
            return lbl
    return "none"


def split_examples(
    examples: list[Example],
    train_frac: float = 0.8,
    dev_frac: float = 0.1,
    seed: int = 13,
) -> dict[str, list[Example]]:
    """Grouped, stratified train/dev/test split.

    Rows sharing a product id always land in the same split (no leakage of
    near-duplicate reviews); groups are stratified by the rarest label they
    contain so thin labels appear in every split.
    """
    groups: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        groups[group_key(ex.row_id)].append(ex)

    strata: dict[str, list[str]] = defaultdict(list)
    for key, exs in groups.items():
        strata[_stratum(exs)].append(key)

    rng = random.Random(seed)
    out: dict[str, list[Example]] = {"train": [], "dev": [], "test": []}
    for stratum in sorted(strata):
        keys = sorted(strata[stratum])
        rng.shuffle(keys)
        total = sum(len(groups[k]) for k in keys)
        train_target = train_frac * total
        dev_target = (train_frac + dev_frac) * total
        seen = 0
        for key in keys:
            if seen < train_target:
                dest = "train"
            elif seen < dev_target:
                dest = "dev"
            else:
                dest = "test"
            out[dest].extend(groups[key])
            seen += len(groups[key])

    for split in out.values():
        split.sort(key=lambda ex: ex.row_id)
    return out


# ---------------------------------------------------------------------------
# Scoring helper (dev-set early stopping; final scoring stays in
# benchmark/llm/evaluate.py)
# ---------------------------------------------------------------------------

def span_micro_f1(
    gold: list[list[dict]],
    pred: list[list[dict]],
) -> tuple[float, float, float]:
    """Micro precision/recall/F1 on exact (label, start, end) span match."""
    tp = fp = fn = 0
    for g_ents, p_ents in zip(gold, pred):
        g = {(e["label"], e["start"], e["end"]) for e in g_ents}
        p = {(e["label"], e["start"], e["end"]) for e in p_ents}
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1
