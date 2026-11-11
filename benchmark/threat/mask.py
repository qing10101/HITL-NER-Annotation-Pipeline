"""Step 1 of the threat experiments: validate offsets, build splits, write masks.

Usage (any Python with pandas):
    python benchmark/threat/mask.py [--detector-pred PRED.csv] [--seed 42]

Outputs under benchmark/threat/data/ (git-ignored; contains review text):
    train.csv / dev.csv / test.csv
        row_id, raw_text, entities_json, minor, gender, kinship,
        gold_del, gold_in, gold_tgt,            # gold spans: deletion / T5 sentinel pair
        rand_del, rand_in, rand_tgt,            # equal-count random word spans
        det_del,  det_in,  det_tgt              # detector spans (test only, if --detector-pred)
    test_loo.csv
        one row per gold span, that span alone masked: row_id, span_idx, label,
        span_text, loo_in, loo_tgt
    lexicon.json
        per-label normalised span frequencies from train+dev
    offset_report.json
"""
from __future__ import annotations

import argparse
import json
import random

import pandas as pd

from common import (CORPUS, DATA_DIR, TEST, build_lexicon, check_offsets,
                    delete_spans, load_entities, presence, random_spans,
                    sentinel_pair)


def add_mask_columns(df: pd.DataFrame, rng: random.Random,
                     det: dict[str, list[dict]] | None = None,
                     content_only: bool = False) -> pd.DataFrame:
    rows = []
    for r in df.itertuples(index=False):
        ents = load_entities(r.entities_json)
        rec = {"row_id": r.row_id, "raw_text": r.raw_text,
               "entities_json": json.dumps(ents), "num_entities": len(ents)}
        rec.update(presence(ents))
        rec["gold_del"] = delete_spans(r.raw_text, ents)
        rec["gold_in"], rec["gold_tgt"] = sentinel_pair(r.raw_text, ents)
        rnd = random_spans(r.raw_text, ents, rng, content_only=content_only)
        rec["rand_json"] = json.dumps(rnd)
        rec["rand_del"] = delete_spans(r.raw_text, rnd)
        rec["rand_in"], rec["rand_tgt"] = sentinel_pair(r.raw_text, rnd)
        if det is not None:
            d = det.get(r.row_id, [])
            rec["det_json"] = json.dumps(d)
            rec["det_del"] = delete_spans(r.raw_text, d)
            rec["det_in"], rec["det_tgt"] = sentinel_pair(r.raw_text, d)
        rows.append(rec)
    return pd.DataFrame(rows)


def loo_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples(index=False):
        ents = load_entities(r.entities_json)
        for i, e in enumerate(ents):
            inp, tgt = sentinel_pair(r.raw_text, [e])
            rows.append({"row_id": r.row_id, "span_idx": i, "label": e["label"],
                         "span_text": e["text"], "loo_in": inp, "loo_tgt": tgt})
    return pd.DataFrame(rows)


def load_detector(path: str) -> dict[str, list[dict]]:
    p = pd.read_csv(path)
    col = "predicted_entities_json" if "predicted_entities_json" in p.columns else "entities_json"
    out = {}
    for rid, ej in zip(p.row_id, p[col]):
        ents = load_entities(ej)
        out[rid] = ents
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--test", default=str(TEST))
    ap.add_argument("--detector-pred", default=None,
                    help="predictions CSV (row_id + predicted_entities_json) for the detector_mask condition")
    ap.add_argument("--dev-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    corpus = pd.read_csv(args.corpus, encoding="utf-8")
    test = pd.read_csv(args.test, encoding="utf-8")
    assert corpus.row_id.is_unique

    # 1. offset validation -----------------------------------------------------
    bad = []
    for r in corpus.itertuples(index=False):
        ents = load_entities(r.entities_json)
        for e in check_offsets(r.raw_text, ents):
            bad.append({"row_id": r.row_id, **e, "actual": r.raw_text[e["start"]:e["end"]]})
    report = {"corpus_rows": len(corpus), "bad_spans": len(bad),
              "bad_rows": len({b["row_id"] for b in bad}), "examples": bad[:20]}
    bad_ids = {b["row_id"] for b in bad}
    if bad_ids:
        print(f"WARNING: dropping {len(bad_ids)} rows with offset mismatches")
        corpus = corpus[~corpus.row_id.isin(bad_ids)]

    # 2. splits ----------------------------------------------------------------
    test_ids = set(test.row_id)
    missing = test_ids - set(corpus.row_id)
    assert not missing, f"{len(missing)} test rows not in corpus"
    test_df = corpus[corpus.row_id.isin(test_ids)].copy()
    pool = corpus[~corpus.row_id.isin(test_ids)].copy()
    rng = random.Random(args.seed)
    pool["_has"] = pool.num_entities > 0
    dev_parts = []
    for has, grp in pool.groupby("_has"):
        ids = list(grp.row_id)
        rng.shuffle(ids)
        dev_parts += ids[: int(round(len(ids) * args.dev_frac))]
    dev_ids = set(dev_parts)
    dev_df = pool[pool.row_id.isin(dev_ids)].drop(columns="_has")
    train_df = pool[~pool.row_id.isin(dev_ids)].drop(columns="_has")

    # 3. masks -----------------------------------------------------------------
    det = load_detector(args.detector_pred) if args.detector_pred else None
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, df, d in [("train", train_df, None), ("dev", dev_df, None), ("test", test_df, det)]:
        m = add_mask_columns(df, random.Random(args.seed + len(name)), d)
        m.to_csv(DATA_DIR / f"{name}.csv", index=False, encoding="utf-8")
        out[name] = m
    loo = loo_rows(test_df)
    loo.to_csv(DATA_DIR / "test_loo.csv", index=False, encoding="utf-8")

    # 4. lexicon from train+dev only -------------------------------------------
    lex = build_lexicon(pd.concat([train_df, dev_df]))
    (DATA_DIR / "lexicon.json").write_text(json.dumps(lex, indent=1, ensure_ascii=False), encoding="utf-8")

    # 5. report ----------------------------------------------------------------
    for name, m in out.items():
        report[name] = {"rows": len(m), "rows_with_span": int((m.num_entities > 0).sum()),
                        "spans": int(m.num_entities.sum()),
                        "presence": {k: int(m[k].sum()) for k in ("minor", "gender", "kinship")}}
    report["test_loo_rows"] = len(loo)
    report["lexicon_sizes"] = {k: len(v) for k, v in lex.items()}
    report["detector_pred"] = args.detector_pred
    (DATA_DIR / "offset_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
