"""Shared helpers for the threat experiments (benchmark/threat/).

Everything here is deterministic and dependency-light (pandas only) so that
mask.py and lexicon.py can run in any environment; the model scripts need
.venv-threat.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

LABELS = ["MINOR_AGE", "MINOR_EDU", "GEN_NOUN", "GEN_PHYS", "FAM_KIN"]
DIMS = {
    "minor": {"MINOR_AGE", "MINOR_EDU"},
    "gender": {"GEN_NOUN", "GEN_PHYS"},
    "kinship": {"FAM_KIN"},
}
SENTINEL = "<extra_id_{}>"

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "gold_standard_merged.csv"
TEST = REPO / "test_balanced_400.csv"
DATA_DIR = Path(__file__).resolve().parent / "data"

# ----------------------------------------------------------------------------
# normalization used by every exact-match comparison
# ----------------------------------------------------------------------------
_WORDNUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
}
_AGE = re.compile(r"^(\d+(?:\.\d+)?)\s*(?:-|\s)?\s*(?:yrs?|years?|yo|y/o|y o)?\s*(?:-|\s)?\s*(?:olds?)?$")
_MONTHS = re.compile(r"^(\d+)\s*(?:-|\s)?\s*(?:m|mo|mos|months?)\s*(?:-|\s)?\s*(?:olds?)?$")


def normalize(span: str) -> str:
    """Lowercase, strip punctuation/possessives, canonicalise ages and months.

    "5 year old", "5yo", "5-year-old", "five year old" -> "age:5"
    "10 month old", "10 months"                          -> "months:10"
    "daughters", "daughter's"                            -> "daughter"
    """
    s = span.lower().strip()
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"'s\b", "", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)          # keep decimal points inside numbers
    s = re.sub(r"[\"',!?;:()\[\]]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = [str(_WORDNUM.get(t, t)) for t in s.split()]
    s = " ".join(toks)
    m = _AGE.match(s)
    if m and re.search(r"\d", s):
        return f"age:{m.group(1).rstrip('0').rstrip('.') if '.' in m.group(1) else m.group(1)}"
    m = _MONTHS.match(s)
    if m:
        return f"months:{m.group(1)}"
    # light lemmatisation of plurals for the kinship/gender nouns
    if len(s) > 4 and s.endswith("ies"):
        s = s[:-3] + "y"
    elif len(s) > 3 and s.endswith("s") and not s.endswith("ss"):
        s = s[:-1]
    return s


# ----------------------------------------------------------------------------
# span utilities
# ----------------------------------------------------------------------------
def load_entities(ej: str) -> list[dict]:
    ents = json.loads(ej) if isinstance(ej, str) else []
    return sorted(ents, key=lambda e: (e["start"], e["end"]))


def check_offsets(text: str, ents: list[dict]) -> list[dict]:
    """Return the spans whose offsets do not reproduce their text."""
    return [e for e in ents if text[e["start"]:e["end"]] != e["text"]]


def presence(ents: list[dict]) -> dict[str, int]:
    labels = {e["label"] for e in ents}
    return {d: int(bool(labels & s)) for d, s in DIMS.items()}


def delete_spans(text: str, ents: list[dict]) -> str:
    """Remove the spans outright and collapse the whitespace left behind."""
    out, pos = [], 0
    for e in ents:
        out.append(text[pos:e["start"]])
        pos = e["end"]
    out.append(text[pos:])
    s = "".join(out)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r" ([.,!?;:])", r"\1", s)
    return s.strip()


def sentinel_pair(text: str, ents: list[dict]) -> tuple[str, str]:
    """T5 span-corruption format: input with <extra_id_k>, target with the spans."""
    inp, tgt, pos = [], [], 0
    for i, e in enumerate(ents):
        inp.append(text[pos:e["start"]])
        inp.append(SENTINEL.format(i))
        tgt.append(SENTINEL.format(i) + " " + e["text"].strip())
        pos = e["end"]
    inp.append(text[pos:])
    tgt.append(SENTINEL.format(len(ents)))
    return "".join(inp), " ".join(tgt)


_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


STOPWORDS = set("""
a an the and or but if so nor yet of to in on at by for with from as into onto about over under than
then too very is am are was were be been being do does did doing have has had having will would shall
should can could may might must not no yes it its it's this that these those there here i me my mine
we us our ours you your yours he him his she her hers they them their theirs who whom whose which what
when where why how all any both each few more most other some such only own same up down out off again
further once also just even still ever never now already because while until before after above below
between through during against s t d ll re ve m br
""".split())


def _is_content(w: str) -> bool:
    return w.lower().strip("'-") not in STOPWORDS and len(w.strip("'-")) > 0


def random_spans(text: str, ents: list[dict], rng: random.Random,
                 n_min: int = 1, n_max: int = 3, content_only: bool = False) -> list[dict]:
    """Pick non-overlapping random word spans that avoid the gold spans.

    With gold spans present: same count and the same word-lengths as gold.
    Without: n in [n_min, n_max] spans of 1-3 words each.
    content_only=True restricts windows to content words (no stopwords), falling
    back to windows whose first and last word are content words, then to any window.
    """
    words = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    if not words:
        return []
    blocked = [(e["start"], e["end"]) for e in ents]

    def free(a, b):
        return all(b <= s or a >= t for s, t in blocked)

    if ents:
        lengths = [max(1, len(_WORD.findall(e["text"]))) for e in ents]
    else:
        lengths = [rng.randint(1, 3) for _ in range(rng.randint(n_min, n_max))]
    picked = []
    toks = [text[a:b] for a, b in words]

    def free_of_picked(a, b):
        return all(b <= s or a >= t for s, t in [(p["start"], p["end"]) for p in picked])

    for L0 in lengths:
        # same word-length as the gold span; if the review has no free window of that
        # length left (short review, many spans), shrink the window rather than drop the
        # span, so every condition keeps the same number of masks per row.
        for L in range(L0, 0, -1):
            cands = [i for i in range(len(words) - L + 1)
                     if free(words[i][0], words[i + L - 1][1])
                     and free_of_picked(words[i][0], words[i + L - 1][1])]
            if not cands:
                continue
            if content_only:
                strict = [i for i in cands if all(_is_content(toks[j]) for j in range(i, i + L))]
                loose = [i for i in cands if _is_content(toks[i]) and _is_content(toks[i + L - 1])]
                cands = strict or loose or cands
            i = rng.choice(cands)
            a, b = words[i][0], words[i + L - 1][1]
            picked.append({"label": "RANDOM", "text": text[a:b], "start": a, "end": b})
            break
    return sorted(picked, key=lambda e: e["start"])


# ----------------------------------------------------------------------------
# lexicon
# ----------------------------------------------------------------------------
def build_lexicon(df: pd.DataFrame, min_count: int = 2) -> dict[str, dict[str, int]]:
    """Per-label frequency table of normalised span strings (training pool only)."""
    cnt: dict[str, Counter] = {l: Counter() for l in LABELS}
    for ej in df.entities_json:
        for e in load_entities(ej):
            cnt[e["label"]][normalize(e["text"])] += 1
    return {l: {k: v for k, v in c.items() if v >= min_count} for l, c in cnt.items()}
