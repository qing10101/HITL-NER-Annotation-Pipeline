"""Context-free reference points for Scheme A: what an attacker scores by always
guessing the most frequent training spans (no reading of the review at all).

    python benchmark/threat/prior_baseline.py   -> results/prior_baseline.json

prior        : top-1 / top-5 normalised spans over all training spans (type unknown)
type-oracle  : top-1 / top-5 within the span's own type (attacker told the type)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, LABELS, load_entities, normalize  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def spans(df: pd.DataFrame) -> list[tuple[str, str]]:
    out = []
    for ej in df[df.num_entities > 0].entities_json:
        out += [(e["label"], normalize(e["text"])) for e in load_entities(ej)]
    return out


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8")
    test = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8")
    lexicon = json.loads((DATA_DIR / "lexicon.json").read_text(encoding="utf-8"))
    tr, te = spans(train), spans(test)
    allc = Counter(n for _, n in tr)
    byl = {l: Counter(n for lab, n in tr if lab == l) for l in LABELS}
    top5 = [w for w, _ in allc.most_common(5)]
    top5_l = {l: [w for w, _ in byl[l].most_common(5)] for l in LABELS}
    print("training span prior, top-5 overall:", top5)
    for l in LABELS:
        print(f"  {l:10} n={sum(byl[l].values()):4d} top-5: {top5_l[l]}")
    res = {"top5_overall": top5, "top5_by_type": top5_l, "test": {}}
    groups = [("all", te)] + [(l, [(a, b) for a, b in te if a == l]) for l in LABELS]
    print(f"\n{'label':10} {'n':>4} | prior exact@1 exact@5 cat@1 | type-oracle exact@1 exact@5")
    for name, g in groups:
        if not g:
            continue
        n = len(g)
        e1 = sum(nm == top5[0] for _, nm in g) / n
        e5 = sum(nm in top5 for _, nm in g) / n
        c1 = sum(top5[0] in lexicon.get(l, {}) for l, _ in g) / n
        o1 = sum(nm == top5_l[l][0] for l, nm in g) / n
        o5 = sum(nm in top5_l[l] for l, nm in g) / n
        res["test"][name] = {"n": n, "prior_exact1": e1, "prior_exact5": e5, "prior_cat1": c1,
                             "oracle_exact1": o1, "oracle_exact5": o5}
        print(f"{name:10} {n:4d} | {e1:14.3f} {e5:8.3f} {c1:6.3f} | {o1:20.3f} {o5:8.3f}")
    (RESULTS / "prior_baseline.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
