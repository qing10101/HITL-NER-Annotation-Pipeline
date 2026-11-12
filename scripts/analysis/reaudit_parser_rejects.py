"""Re-audit the rows the parser rejects, with the current auditor prompt.

The parser rejects a row when the tag-stripped annotator output does not
reproduce the raw text (strip(ỹ) != x) or a tag is malformed. This script sends
those rows' cached annotator output ỹ back to the pipeline's auditor (the
current AUDITOR_SYSTEM_PROMPT, $AUDITOR_MODEL, $AUDITOR_TEMPERATURE) and
compares the new verdicts with the ones recorded in the run's run_log.csv:
does the auditor now name RAW_TEXT_MUTATION, or still a different error type?

Rows with no annotator output (content-blocked) have nothing to audit and are
skipped. Each row is audited --repeats times, since the auditor samples at a
non-zero temperature.

Examples
--------
  # balanced 400 set (default)
  python scripts/analysis/reaudit_parser_rejects.py

  # random 500 set
  python scripts/analysis/reaudit_parser_rejects.py \\
      --human-csv validation/general/sample_500_human.csv \\
      --out-csv validation/general/reaudit_parser_rejects.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import ablation_c1_c3 as abl  # noqa: E402
from pipeline.config import CONFIG  # noqa: E402

RUN_DIR = ROOT / "output"
HUMAN_CSV = ROOT / "validation" / "test_balanced_400_human_validated.csv"
SOURCE_CSV = RUN_DIR / "gold_standard_merged.csv"
OUT_CSV = ROOT / "validation" / "test_balanced_400_reaudit_parser_rejects.csv"

_HEADER = ["row_id", "repeat", "old_status", "old_error_type",
           "new_status", "new_error_type", "new_auditor_reason"]


async def _main(args: argparse.Namespace) -> None:
    from pipeline.auditor import Auditor

    CONFIG.require_keys([CONFIG.auditor_model])
    auditor = Auditor()

    ids = sorted(r["row_id"] for r in abl._read(args.human_csv))
    raw = {r["row_id"]: r["raw_text"] for r in abl._read(args.source_csv)}
    gemini = {r["row_id"]: r["tagged_text"] for r in abl._read(args.run_dir / "annotator_cache.csv")}
    log = {r["row_id"]: r for r in abl._read(args.run_dir / "run_log.csv")}

    rejected = [rid for rid in ids if rid in gemini and abl._parser_rejects(gemini[rid], raw[rid])]
    blocked = [rid for rid in ids if rid not in gemini]
    print(f"Auditor: {CONFIG.auditor_model} (temperature={auditor.temperature})")
    print(f"Rows: {len(ids)}  parser rejects: {len(rejected)}  "
          f"no annotator output (skipped): {len(blocked)}  repeats: {args.repeats}\n")

    async def _one(rid: str, k: int) -> list:
        try:
            res = await auditor.audit_async(raw[rid], gemini[rid])
            new = [res.status.value, res.error_type.value, res.auditor_reason]
        except Exception as exc:  # noqa: BLE001 - isolate per-row failures
            new = ["FAIL", "PIPELINE_ERROR", f"{type(exc).__name__}: {exc}"]
        return [rid, k, log[rid]["status"], log[rid]["error_type"], *new]

    results = await asyncio.gather(*(_one(rid, k) for rid in rejected for k in range(1, args.repeats + 1)))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_HEADER)
        w.writerows(results)

    print(f"{'Row':<20} {'Old error type':<23} New error types ({args.repeats} audits)")
    for rid in rejected:
        new = Counter(r[5] for r in results if r[0] == rid)
        print(f"{rid:<20} {log[rid]['error_type']:<23} "
              + ", ".join(f"{et} ×{n}" for et, n in new.most_common()))
    mutation = sum(r[5] == "RAW_TEXT_MUTATION" for r in results)
    print(f"\nRAW_TEXT_MUTATION: {mutation}/{len(results)} audits "
          f"(old prompt: {sum(log[rid]['error_type'] == 'RAW_TEXT_MUTATION' for rid in rejected)}"
          f"/{len(rejected)} rows)")
    print(f"Written: {args.out_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, default=RUN_DIR,
                   help="Run with annotator_cache.csv and run_log.csv")
    p.add_argument("--human-csv", type=Path, default=HUMAN_CSV, help="Evaluation set (row_id column)")
    p.add_argument("--source-csv", type=Path, default=SOURCE_CSV, help="Raw text x by row_id")
    p.add_argument("--out-csv", type=Path, default=OUT_CSV)
    p.add_argument("--repeats", type=int, default=3, help="Audits per row")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(_main(parse_args()))
