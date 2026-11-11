#!/bin/bash
# lists batches whose reply is missing or not valid JSON with the expected ids
cd "$(dirname "$0")/../../.." || exit 1
.venv-threat/Scripts/python - <<'PY'
import json, sys
from pathlib import Path
B=Path("benchmark/threat/llm/data/batches"); R=Path("benchmark/threat/results/recon_claude-fable-5-1_subagent_s0/raw_batches")
pending=[]; ok=0
for cond in ("gold","loo","rand","rand_s43","rand_s44","memcheck"):
    m=json.loads((B/cond/"manifest.json").read_text(encoding="utf-8"))
    for i,ids in enumerate(m["ids"]):
        f=R/cond/f"batch_{i:03d}.json"
        if not f.exists(): pending.append(f"{cond}/batch_{i:03d} missing"); continue
        try:
            t=f.read_text(encoding="utf-8").strip()
            if t.startswith("```"): t=t.strip("`").split("\n",1)[1].rsplit("```",1)[0]
            got={str(e.get("id")) for e in json.loads(t)["items"]}
            miss=set(ids)-got
            if miss: pending.append(f"{cond}/batch_{i:03d} lacks {len(miss)} ids"); 
            else: ok+=1
        except Exception as e: pending.append(f"{cond}/batch_{i:03d} bad json: {e}")
print(f"ok={ok} pending={len(pending)}")
for p in pending: print(p)
PY
