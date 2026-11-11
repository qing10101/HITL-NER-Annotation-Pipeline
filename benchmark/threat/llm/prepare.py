"""Step 1 of the LLM reconstruction attack: build every input file from the
human-validated test set.

    python benchmark/threat/llm/prepare.py [--seeds 42 43 44] [--memcheck-n 50]

Reads  <repo>/test_balanced_400_human_validated - test_balanced_400.csv
       (row_id, raw_text, tagged_text; spans are <LABEL>...</LABEL> in tagged_text)

Writes benchmark/threat/data/test.csv, test_loo.csv   (shared with recon_t5.py, so the
       T5 attackers can be re-scored on exactly the same spans; the previous
       pipeline-labelled files are kept as test_pipeline.csv / test_loo_pipeline.csv)
       benchmark/threat/llm/data/
         gold.jsonl         300 prompts, 501 masks  (all gold spans masked)
         loo.jsonl          320 prompts, 1 mask each (multi-span rows only; the 181
                            single-span rows are identical to gold and are re-used)
         loo_reuse.json     the (row_id, span_idx) pairs re-used from gold
         rand_s<seed>.jsonl 300 prompts, 501 masks of length-matched content words
         memcheck.jsonl     continuation probes for the memorisation check
         manifest.json      counts

Mask token: [MASK_k], one per span regardless of its length, numbered left to right.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
THREAT = HERE.parent
sys.path.insert(0, str(THREAT))
from common import DATA_DIR, LABELS, load_entities, normalize, random_spans  # noqa: E402
from mask import add_mask_columns, loo_rows  # noqa: E402

REPO = THREAT.parents[1]
VALIDATED = REPO / "test_balanced_400_human_validated - test_balanced_400.csv"
OUT = HERE / "data"
TAG = re.compile(r"<(FAM_KIN|MINOR_AGE|GEN_NOUN|GEN_PHYS|MINOR_EDU)>(.*?)</\1>", re.S)
SENT = re.compile(r"[^.!?]+(?:[.!?]+|$)")


def parse_tagged(tagged: str, raw: str) -> list[dict]:
    """Entities with character offsets into raw, derived from the tagged text."""
    ents, plain, pos = [], [], 0
    for m in TAG.finditer(tagged):
        plain.append(tagged[pos:m.start()])
        start = sum(len(p) for p in plain)
        plain.append(m.group(2))
        ents.append({"label": m.group(1), "text": m.group(2), "start": start, "end": start + len(m.group(2))})
        pos = m.end()
    plain.append(tagged[pos:])
    assert "".join(plain) == raw, "untagged text does not reproduce raw_text"
    for e in ents:
        assert raw[e["start"]:e["end"]] == e["text"]
    return ents


def mask_text(text: str, ents: list[dict]) -> str:
    out, pos = [], 0
    for k, e in enumerate(sorted(ents, key=lambda e: e["start"]), 1):
        out.append(text[pos:e["start"]])
        out.append(f"[MASK_{k}]")
        pos = e["end"]
    out.append(text[pos:])
    return "".join(out)


def item(row_id: str, cond: str, seed: int | None, text: str, ents: list[dict],
         item_id: str | None = None, span_idx: list[int] | None = None) -> dict:
    ents = sorted(ents, key=lambda e: e["start"])
    return {"id": item_id or row_id, "row_id": row_id, "cond": cond, "seed": seed,
            "masked_text": mask_text(text, ents),
            "spans": [{"k": k, "span_idx": (span_idx[k - 1] if span_idx else k - 1),
                       "label": e["label"], "text": e["text"], "norm": normalize(e["text"])}
                      for k, e in enumerate(ents, 1)]}


def write_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def memcheck_items(df: pd.DataFrame, n: int, rng: random.Random) -> list[dict]:
    """Prefix (gold-masked) + the next raw sentence, which must contain no gold span."""
    out = []
    rows = list(df.itertuples(index=False))
    rng.shuffle(rows)
    for r in rows:
        ents = load_entities(r.entities_json)
        sents = [(m.start(), m.end()) for m in SENT.finditer(r.raw_text) if m.group().strip()]
        if len(sents) < 3:
            continue
        choices = [i for i in range(2, len(sents))
                   if not any(e["start"] < sents[i][1] and e["end"] > sents[i][0] for e in ents)
                   and len(r.raw_text[sents[i][0]:sents[i][1]].split()) >= 5]
        if not choices:
            continue
        i = rng.choice(choices)
        cut = sents[i][0]
        prefix_ents = [e for e in ents if e["end"] <= cut]
        out.append({"id": r.row_id, "row_id": r.row_id,
                    "prefix_masked": mask_text(r.raw_text[:cut], prefix_ents),
                    "target": r.raw_text[sents[i][0]:sents[i][1]].strip()})
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validated", default=str(VALIDATED))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--memcheck-n", type=int, default=50)
    args = ap.parse_args()

    hv = pd.read_csv(args.validated, encoding="utf-8")
    assert hv.row_id.is_unique
    recs = []
    for r in hv.itertuples(index=False):
        ents = parse_tagged(r.tagged_text, r.raw_text)
        recs.append({"row_id": r.row_id, "raw_text": r.raw_text,
                     "entities_json": json.dumps(ents, ensure_ascii=False), "num_entities": len(ents)})
    test_df = pd.DataFrame(recs)

    # 1. shared test split (keeps recon_t5.py / prior_baseline.py on the same spans) -------
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("test", "test_loo"):
        src, bak = DATA_DIR / f"{name}.csv", DATA_DIR / f"{name}_pipeline.csv"
        if src.exists() and not bak.exists():
            shutil.copy(src, bak)
    rng42 = random.Random(42 + len("test"))  # same seed mask.py used for the test split
    test_m = add_mask_columns(test_df, rng42, None, content_only=True)
    test_m.to_csv(DATA_DIR / "test.csv", index=False, encoding="utf-8")
    loo = loo_rows(test_df)
    loo.to_csv(DATA_DIR / "test_loo.csv", index=False, encoding="utf-8")

    # 2. LLM condition files ----------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    pos = test_m[test_m.num_entities > 0].reset_index(drop=True)
    gold_items, loo_items, reuse = [], [], []
    for r in pos.itertuples(index=False):
        ents = load_entities(r.entities_json)
        gold_items.append(item(r.row_id, "gold", None, r.raw_text, ents))
        if len(ents) == 1:
            reuse.append({"row_id": r.row_id, "span_idx": 0})
        else:
            for i, e in enumerate(ents):
                loo_items.append(item(r.row_id, "loo", None, r.raw_text, [e],
                                      item_id=f"{r.row_id}#{i}", span_idx=[i]))
    write_jsonl(OUT / "gold.jsonl", gold_items)
    write_jsonl(OUT / "loo.jsonl", loo_items)
    (OUT / "loo_reuse.json").write_text(json.dumps(reuse, indent=1), encoding="utf-8")

    rand_counts = {}
    for seed in args.seeds:
        items = []
        if seed == 42:  # identical to the rand_json column recon_t5.py scores
            for r in pos.itertuples(index=False):
                items.append(item(r.row_id, "rand", seed, r.raw_text, json.loads(r.rand_json)))
        else:
            rng = random.Random(seed)
            for r in pos.itertuples(index=False):
                rnd = random_spans(r.raw_text, load_entities(r.entities_json), rng, content_only=True)
                items.append(item(r.row_id, "rand", seed, r.raw_text, rnd))
        write_jsonl(OUT / f"rand_s{seed}.jsonl", items)
        rand_counts[seed] = sum(len(it["spans"]) for it in items)

    mem = memcheck_items(pos, args.memcheck_n, random.Random(42))
    write_jsonl(OUT / "memcheck.jsonl", mem)

    # 3. manifest ---------------------------------------------------------------------
    gold_spans = [s for it in gold_items for s in it["spans"]]
    manifest = {
        "source": str(Path(args.validated).name),
        "rows_total": len(test_m), "rows_positive": len(pos), "rows_negative_dropped": int((test_m.num_entities == 0).sum()),
        "spans": len(gold_spans), "spans_by_label": dict(Counter(s["label"] for s in gold_spans)),
        "span_word_lengths": dict(Counter(len(s["text"].split()) for s in gold_spans)),
        "gold_prompts": len(gold_items),
        "loo_prompts": len(loo_items), "loo_reused_from_gold": len(reuse),
        "loo_spans_total": len(loo_items) + len(reuse),
        "rand_spans_by_seed": rand_counts,
        "memcheck_items": len(mem),
        "mask_token": "[MASK_k]",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    assert manifest["loo_spans_total"] == manifest["spans"] == rand_counts[args.seeds[0]], "span counts differ"


if __name__ == "__main__":
    main()
