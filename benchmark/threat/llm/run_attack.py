"""Zero-shot LLM reconstruction attacker (Scheme A) via the Anthropic API.

    .venv-threat/Scripts/python benchmark/threat/llm/run_attack.py --cond gold [--limit 20]
    .venv-threat/Scripts/python benchmark/threat/llm/run_attack.py --cond loo
    .venv-threat/Scripts/python benchmark/threat/llm/run_attack.py --cond rand --seed 42
    .venv-threat/Scripts/python benchmark/threat/llm/run_attack.py --cond rand --seed 43

Reads  benchmark/threat/llm/data/<cond>[_s<seed>].jsonl  (from prepare.py)
Writes benchmark/threat/results/recon_<model>_<tag>_s0/
         raw/<cond>/<id>.json      every request/response, so a rerun resumes and nothing is re-billed
         preds_<cond>.csv          one row per masked span, same columns as recon_t5.py
         metrics.json              per condition x label (score.py)

Every item is one independent API call with a fresh context: the model never sees
another review, a gold label, or this repository. The prompt is policy-blind
(prompts.py). Sampling parameters are not accepted by current models, so runs are
not bit-for-bit deterministic; the raw cache fixes the predictions that were scored.

Credentials: ANTHROPIC_API_KEY in the environment or in <repo>/.env.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
THREAT = HERE.parent
sys.path.insert(0, str(THREAT))
sys.path.insert(0, str(HERE))
from common import DATA_DIR, LABELS, normalize  # noqa: E402
from prompts import SCHEMA, SYSTEM, user_prompt  # noqa: E402
from score import write_metrics  # noqa: E402

REPO = THREAT.parents[1]
RESULTS = THREAT / "results"
LLM_DATA = HERE / "data"
N_CANDS = 5


def load_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env")
    except ImportError:
        pass
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set (environment or <repo>/.env)")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def safe_name(item_id: str) -> str:
    return item_id.replace("#", "_")


def parse_guesses(text: str) -> dict[int, dict]:
    data = json.loads(text)
    out = {}
    for g in data.get("guesses", []):
        try:
            k = int(g["mask"])
        except (KeyError, TypeError, ValueError):
            continue
        cands = [str(c).strip() for c in g.get("candidates", []) if str(c).strip()]
        cands = (cands + [""] * N_CANDS)[:N_CANDS]
        try:
            conf = min(1.0, max(0.0, float(g.get("confidence", 0.0))))
        except (TypeError, ValueError):
            conf = 0.0
        out[k] = {"candidates": cands, "confidence": conf}
    return out


def call_one(client, item: dict, args, cache_path: Path) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    import anthropic
    rec = {"id": item["id"], "row_id": item["row_id"], "cond": item["cond"], "seed": item.get("seed"),
           "model_requested": args.model, "effort": args.effort, "ts": time.time()}
    prompt = user_prompt(item["masked_text"], len(item["spans"]))
    try:
        resp = client.messages.create(
            model=args.model,
            max_tokens=args.max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": args.effort,
                           "format": {"type": "json_schema", "schema": SCHEMA}},
        )
    except anthropic.APIStatusError as e:  # 4xx that the SDK did not retry away
        rec.update({"error": f"{e.status_code}: {e.message}", "stop_reason": "error", "guesses": {}})
        cache_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        return rec
    rec["model"] = resp.model
    rec["stop_reason"] = resp.stop_reason
    rec["usage"] = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    text = next((b.text for b in resp.content if b.type == "text"), "")
    rec["text"] = text
    if resp.stop_reason == "refusal":
        sd = getattr(resp, "stop_details", None)
        rec["refusal"] = {"category": getattr(sd, "category", None), "explanation": getattr(sd, "explanation", None)}
        rec["guesses"] = {}
    else:
        try:
            rec["guesses"] = parse_guesses(text)
        except (json.JSONDecodeError, AttributeError) as e:
            rec["error"] = f"parse: {e}"
            rec["guesses"] = {}
    cache_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


def score_items(items: list[dict], recs: dict[str, dict], lexicon: dict) -> pd.DataFrame:
    any_lex = set().union(*(set(v) for v in lexicon.values()))
    rows = []
    for it in items:
        rec = recs[it["id"]]
        guesses = {int(k): v for k, v in rec.get("guesses", {}).items()}
        for s in it["spans"]:
            g = guesses.get(s["k"], {"candidates": [""] * N_CANDS, "confidence": 0.0})
            preds = g["candidates"]
            preds_n = [normalize(p) if p else "" for p in preds]
            lab, gold_n = s["label"], s["norm"]
            lex = set(lexicon.get(lab, {}))
            rows.append({
                "row_id": it["row_id"], "span_idx": s["span_idx"], "label": lab,
                "gold": s["text"], "gold_norm": gold_n,
                **{f"pred{j + 1}": preds[j] for j in range(N_CANDS)},
                "exact1": int(bool(preds_n[0]) and preds_n[0] == gold_n),
                "exact5": int(gold_n in [p for p in preds_n if p]),
                "cat1": (int(preds_n[0] in lex) if lab in LABELS else None),
                "anycat1": int(preds_n[0] in any_lex),
                "conf1": g["confidence"],
                "gold_prob": None,  # teacher-forced probability is not observable through the API
                "stop_reason": rec.get("stop_reason"),
                "item_id": it["id"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=["gold", "loo", "rand"], required=True)
    ap.add_argument("--seed", type=int, default=42, help="random-mask seed (rand only)")
    ap.add_argument("--model", default="claude-fable-5-1")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--tag", default="zeroshot")
    ap.add_argument("--limit", type=int, default=None, help="pilot: first N prompts only")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and exit")
    args = ap.parse_args()

    fname = f"{args.cond}_s{args.seed}.jsonl" if args.cond == "rand" else f"{args.cond}.jsonl"
    items = read_jsonl(LLM_DATA / fname)
    if args.limit:
        items = items[: args.limit]
    cond_key = f"rand_s{args.seed}" if (args.cond == "rand" and args.seed != 42) else args.cond

    if args.dry_run:
        print("SYSTEM:\n" + SYSTEM + "\n\nUSER:\n" + user_prompt(items[0]["masked_text"], len(items[0]["spans"])))
        print("\nGOLD:", [(s["k"], s["label"], s["text"]) for s in items[0]["spans"]])
        return

    load_key()
    import anthropic
    client = anthropic.Anthropic(max_retries=6, timeout=300)
    out_dir = RESULTS / f"recon_{args.model}_{args.tag}_s0"
    raw_dir = out_dir / "raw" / cond_key
    raw_dir.mkdir(parents=True, exist_ok=True)

    recs, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(call_one, client, it, args, raw_dir / f"{safe_name(it['id'])}.json"): it for it in items}
        for n, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            recs[rec["id"]] = rec
            if n % 25 == 0 or n == len(items):
                print(f"  {n}/{len(items)} done, {time.time() - t0:.0f}s", flush=True)

    lexicon = json.loads((DATA_DIR / "lexicon.json").read_text(encoding="utf-8"))
    pred = score_items(items, recs, lexicon)

    if args.cond == "loo" and not args.limit:
        gold_path = out_dir / "preds_gold.csv"
        if not gold_path.exists():
            sys.exit("run --cond gold first: single-span rows are re-used from preds_gold.csv")
        gold = pd.read_csv(gold_path, encoding="utf-8")
        reuse = pd.DataFrame(json.loads((LLM_DATA / "loo_reuse.json").read_text(encoding="utf-8")))
        reused = gold.merge(reuse, on=["row_id", "span_idx"])
        reused["item_id"] = "reuse:gold"
        pred = pd.concat([pred, reused], ignore_index=True)

    stem = f"preds_{cond_key}" + ("_pilot" if args.limit else "")
    pred.to_csv(out_dir / f"{stem}.csv", index=False, encoding="utf-8")

    usage = [r.get("usage") for r in recs.values() if r.get("usage")]
    n_ref = sum(r.get("stop_reason") == "refusal" for r in recs.values())
    n_err = sum("error" in r for r in recs.values())
    print(f"\n{cond_key}: {len(items)} prompts, {len(pred)} spans scored; refusals={n_ref} errors={n_err}")
    if usage:
        print(f"tokens: in={sum(u['input_tokens'] for u in usage)} out={sum(u['output_tokens'] for u in usage)}")
    for lab, g in [("all", pred)] + [(l, pred[pred.label == l]) for l in sorted(pred.label.unique())]:
        print(f"  {lab:10} n={len(g):4d} exact@1={g.exact1.mean():.3f} exact@5={g.exact5.mean():.3f} "
              f"anycat@1={g.anycat1.mean():.3f} conf1={g.conf1.mean():.3f}")
    if not args.limit:
        write_metrics(out_dir)


if __name__ == "__main__":
    main()
