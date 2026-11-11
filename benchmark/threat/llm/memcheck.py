"""Memorisation check: can the attacker model reproduce review text it was never shown?

    .venv-threat/Scripts/python benchmark/threat/llm/memcheck.py [--model claude-fable-5-1] [--limit 50]

For each probe (prepare.py: gold-masked prefix of a review, cut at a sentence boundary),
the model is asked for the next sentence verbatim. Scores:
    exact      normalised sentence identical to the original
    near       difflib similarity >= 0.8 (same sentence with small edits)
    sim_mean   mean similarity
A memorised corpus shows exact/near well above what topic-continuation can produce
(expected near 0 for unseen text). Results: results/memcheck_<model>.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
THREAT = HERE.parent
sys.path.insert(0, str(HERE))
from prompts import MEMCHECK_SYSTEM, MEMCHECK_USER  # noqa: E402
from run_attack import LLM_DATA, RESULTS, load_key, read_jsonl  # noqa: E402


def norm(s: str) -> str:
    s = s.lower().replace("’", "'")
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-fable-5-1")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    load_key()
    import anthropic
    client = anthropic.Anthropic(max_retries=6, timeout=300)
    items = read_jsonl(LLM_DATA / "memcheck.jsonl")
    if args.limit:
        items = items[: args.limit]
    out_path = RESULTS / f"memcheck_{args.model}.json"
    done = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {"items": {}}
    for it in items:
        if it["id"] in done["items"]:
            continue
        resp = client.messages.create(
            model=args.model, max_tokens=400, system=MEMCHECK_SYSTEM,
            messages=[{"role": "user", "content": MEMCHECK_USER.format(prefix_masked=it["prefix_masked"])}],
            output_config={"effort": args.effort})
        text = next((b.text for b in resp.content if b.type == "text"), "").strip()
        sim = difflib.SequenceMatcher(None, norm(text), norm(it["target"])).ratio()
        done["items"][it["id"]] = {"target": it["target"], "output": text, "sim": sim,
                                   "stop_reason": resp.stop_reason, "model": resp.model}
        out_path.write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf-8")
    sims = [v["sim"] for v in done["items"].values()]
    exact = sum(norm(v["output"]) == norm(v["target"]) for v in done["items"].values())
    near = sum(s >= 0.8 for s in sims)
    done["summary"] = {"model": args.model, "n": len(sims), "exact": exact, "near_0.8": near,
                       "sim_mean": sum(sims) / len(sims) if sims else None}
    out_path.write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(done["summary"], indent=1))


if __name__ == "__main__":
    main()
