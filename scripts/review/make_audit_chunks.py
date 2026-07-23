"""Emit clean, human/agent-readable audit chunks from the pipeline outputs.

Each row is rendered as RAW vs TAGGED plus the auditor's verdict, so an
independent reviewer can re-judge it against the guideline. Splits into N files.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

D = Path("output/amazon_unknown_100")
OUT = Path("output/_audit")
OUT.mkdir(parents=True, exist_ok=True)
CHUNKS = 5

rows = []

with (D / "gold_standard.csv").open(encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append({
            "row_id": r["row_id"],
            "verdict": "PASS",
            "error_type": "NONE",
            "reason": "",
            "raw": r["raw_text"],
            "tagged": r["tagged_text"],
            "n": r["num_entities"],
        })

with (D / "review_queue.csv").open(encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append({
            "row_id": r["row_id"],
            "verdict": "FAIL",
            "error_type": r["error_type"],
            "reason": r["auditor_reason"],
            "raw": r["raw_text"],
            "tagged": r["tagged_text"],
            "n": "?",
        })

per = math.ceil(len(rows) / CHUNKS)
for ci in range(CHUNKS):
    block = rows[ci * per:(ci + 1) * per]
    if not block:
        continue
    lines = []
    for r in block:
        verdict = f"AUDITOR={r['verdict']}"
        if r["verdict"] == "FAIL":
            verdict += f" ({r['error_type']}: {r['reason']})"
        lines.append(f"### {r['row_id']}  [{verdict}]")
        lines.append(f"RAW:    {r['raw']}")
        lines.append(f"TAGGED: {r['tagged']}")
        lines.append("")
    path = OUT / f"chunk_{ci + 1}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path} ({len(block)} rows)")

print(f"total rows: {len(rows)}")
