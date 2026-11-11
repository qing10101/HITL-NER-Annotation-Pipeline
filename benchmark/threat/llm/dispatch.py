"""Subagent dispatch path for the LLM reconstruction attack (no API key needed).

    python benchmark/threat/llm/dispatch.py make    --cond gold|loo|rand|memcheck [--seed 42] [--batch 15]
    python benchmark/threat/llm/dispatch.py collect --cond gold|loo|rand|memcheck [--seed 42] [--run-dir NAME]

`make`    writes self-contained batch prompts to data/batches/<cond>/batch_NNN.txt. Each
          batch is handed verbatim to one fresh subagent, whose reply (JSON only) is
          saved to results/<run-dir>/raw_batches/<cond>/batch_NNN.json.
`collect` parses the saved replies into preds_<cond>.csv (same columns as run_attack.py),
          applies the LOO re-use of single-span rows, and refreshes metrics.json.

Batch rules (leak prevention):
  * one condition per batch;
  * LOO: at most one item per review per batch (two LOO items of the same review would
    reveal each other's span);
  * a review's gold/LOO prompt never shares a batch with its random-mask prompt
    (the random prompt shows the gold spans in clear) - guaranteed by rule 1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
THREAT = HERE.parent
sys.path.insert(0, str(THREAT))
sys.path.insert(0, str(HERE))
from common import DATA_DIR  # noqa: E402
from prompts import MEMCHECK_SYSTEM, SYSTEM, user_prompt  # noqa: E402
from run_attack import LLM_DATA, N_CANDS, RESULTS, parse_guesses, read_jsonl, score_items  # noqa: E402
from score import write_metrics  # noqa: E402

BATCH_DIR = LLM_DATA / "batches"

HEADER = (
    "TASK. Reply with JSON only (no prose, no code fence). Do not use any tools, do not read "
    "any file, do not search: everything you need is in this message.\n\n"
    "{system}\n\n"
    "You will receive {n} independent items. Treat each item on its own. Output exactly this "
    "JSON shape, one entry per item id, in the same order:\n"
    '{{"items": [{{"id": "<item id>", "guesses": [{{"mask": 1, "candidates": ["g1","g2","g3","g4","g5"], '
    '"confidence": 0.0}}, ...]}}, ...]}}\n\n'
)
MEM_HEADER = (
    "TASK. Reply with JSON only (no prose, no code fence). Do not use any tools, do not read "
    "any file, do not search: everything you need is in this message.\n\n"
    "{system}\n\n"
    "You will receive {n} independent items. Output exactly: "
    '{{"items": [{{"id": "<item id>", "next_sentence": "..."}}, ...]}}\n\n'
)


def loo_batches(items: list[dict], size: int) -> list[list[dict]]:
    batches: list[list[dict]] = []
    for it in items:
        for b in batches:
            if len(b) < size and all(x["row_id"] != it["row_id"] for x in b):
                b.append(it)
                break
        else:
            batches.append([it])
    return batches


def make(args) -> None:
    if args.cond == "memcheck":
        items = read_jsonl(LLM_DATA / "memcheck.jsonl")
        tpl, sys_text = MEM_HEADER, MEMCHECK_SYSTEM
    else:
        fname = f"rand_s{args.seed}.jsonl" if args.cond == "rand" else f"{args.cond}.jsonl"
        items = read_jsonl(LLM_DATA / fname)
        tpl, sys_text = HEADER, SYSTEM
    batches = loo_batches(items, args.batch) if args.cond == "loo" else \
        [items[i:i + args.batch] for i in range(0, len(items), args.batch)]
    key = cond_key(args)
    out = BATCH_DIR / key
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("batch_*.txt"):
        f.unlink()
    for i, b in enumerate(batches):
        parts = [tpl.format(system=sys_text, n=len(b))]
        for it in b:
            if args.cond == "memcheck":
                parts.append(f"### Item {it['id']}\nBeginning of the review:\n\n{it['prefix_masked']}\n\nNext sentence:\n")
            else:
                parts.append(f"### Item {it['id']}\n{user_prompt(it['masked_text'], len(it['spans']))}\n")
        (out / f"batch_{i:03d}.txt").write_text("\n".join(parts), encoding="utf-8")
    manifest = {"cond": key, "items": len(items), "batches": len(batches), "batch_size": args.batch,
                "ids": [[it["id"] for it in b] for b in batches]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{key}: {len(items)} items -> {len(batches)} batches in {out}")


def cond_key(args) -> str:
    if args.cond == "rand" and args.seed != 42:
        return f"rand_s{args.seed}"
    return args.cond


def collect(args) -> None:
    key = cond_key(args)
    run_dir = RESULTS / args.run_dir
    raw = run_dir / "raw_batches" / key
    manifest = json.loads((BATCH_DIR / key / "manifest.json").read_text(encoding="utf-8"))
    replies: dict[str, dict] = {}
    missing = []
    for i in range(manifest["batches"]):
        f = raw / f"batch_{i:03d}.json"
        if not f.exists():
            missing.append(f.name)
            continue
        text = f.read_text(encoding="utf-8").strip()
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[1] if "\n" in text else text
            text = text.rsplit("```", 1)[0]
        data = json.loads(text)
        for ent in data.get("items", []):
            replies[str(ent.get("id"))] = ent
    if missing:
        print(f"WARNING: {len(missing)} batch replies missing: {missing[:5]}{' ...' if len(missing) > 5 else ''}")

    if args.cond == "memcheck":
        from memcheck import norm
        import difflib
        items = read_jsonl(LLM_DATA / "memcheck.jsonl")
        res = {"items": {}}
        for it in items:
            out = str(replies.get(it["id"], {}).get("next_sentence", ""))
            sim = difflib.SequenceMatcher(None, norm(out), norm(it["target"])).ratio()
            res["items"][it["id"]] = {"target": it["target"], "output": out, "sim": sim}
        sims = [v["sim"] for v in res["items"].values()]
        res["summary"] = {"n": len(sims), "exact": sum(norm(v["output"]) == norm(v["target"]) for v in res["items"].values()),
                          "near_0.8": sum(s >= 0.8 for s in sims), "sim_mean": sum(sims) / len(sims) if sims else None,
                          "answered": sum(bool(v["output"]) for v in res["items"].values())}
        (run_dir / "memcheck.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(res["summary"], indent=1))
        return

    fname = f"rand_s{args.seed}.jsonl" if args.cond == "rand" else f"{args.cond}.jsonl"
    items = read_jsonl(LLM_DATA / fname)
    recs = {}
    for it in items:
        ent = replies.get(it["id"])
        if ent is None:
            recs[it["id"]] = {"id": it["id"], "stop_reason": "missing", "guesses": {}}
            continue
        try:
            guesses = parse_guesses(json.dumps({"guesses": ent.get("guesses", [])}))
            recs[it["id"]] = {"id": it["id"], "stop_reason": "end_turn", "guesses": guesses}
        except Exception as e:  # noqa: BLE001
            recs[it["id"]] = {"id": it["id"], "stop_reason": f"parse:{e}", "guesses": {}}
    lexicon = json.loads((DATA_DIR / "lexicon.json").read_text(encoding="utf-8"))
    pred = score_items(items, recs, lexicon)
    if args.cond == "loo":
        gold_path = run_dir / "preds_gold.csv"
        if not gold_path.exists():
            sys.exit("collect gold first: single-span rows are re-used from preds_gold.csv")
        gold = pd.read_csv(gold_path, encoding="utf-8")
        reuse = pd.DataFrame(json.loads((LLM_DATA / "loo_reuse.json").read_text(encoding="utf-8")))
        reused = gold.merge(reuse, on=["row_id", "span_idx"]).assign(item_id="reuse:gold")
        pred = pd.concat([pred, reused], ignore_index=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(run_dir / f"preds_{key}.csv", index=False, encoding="utf-8")
    n_missing = int((pred.stop_reason != "end_turn").sum())
    print(f"{key}: {len(pred)} spans scored, {n_missing} unanswered")
    for lab, g in [("all", pred)] + [(l, pred[pred.label == l]) for l in sorted(pred.label.unique())]:
        print(f"  {lab:10} n={len(g):4d} exact@1={g.exact1.mean():.3f} exact@5={g.exact5.mean():.3f} "
              f"anycat@1={g.anycat1.mean():.3f} conf1={g.conf1.mean():.3f}")
    write_metrics(run_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["make", "collect"])
    ap.add_argument("--cond", choices=["gold", "loo", "rand", "memcheck"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=15)
    ap.add_argument("--run-dir", default="recon_claude-fable-5-1_subagent_s0")
    args = ap.parse_args()
    (make if args.action == "make" else collect)(args)


if __name__ == "__main__":
    main()
