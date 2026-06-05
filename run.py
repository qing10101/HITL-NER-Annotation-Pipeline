"""CLI entrypoint for the Cascading Multi-Agent Inline Tagging & Audit pipeline.

Examples
--------
  python run.py --input data/sample_reviews.jsonl
  python run.py --input data/reviews.csv --text-field review --limit 500
  python run.py --input data/reviews.jsonl --out-dir output/run1 --no-resume
"""
from __future__ import annotations

import argparse
import sys
import time

from config import CONFIG
from pipeline.ingestion import stream_rows
from pipeline.orchestrator import Orchestrator
from pipeline.writers import OutputWriter


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Privacy-NER multi-agent labeling pipeline")
    p.add_argument("--input", required=True, help="Path to input .jsonl or .csv")
    p.add_argument("--text-field", default="text", help="Field/column with review text")
    p.add_argument("--id-field", default="id", help="Field/column with a stable row id")
    p.add_argument(
        "--format",
        default="auto",
        choices=["auto", "jsonl", "csv"],
        help="Input format (default: inferred from extension)",
    )
    p.add_argument("--out-dir", default=CONFIG.output_dir, help="Output directory")
    p.add_argument("--limit", type=int, default=None, help="Max rows to process")
    p.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between rows (throttle for rate-limited tiers)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocess rows already present in run_log.csv",
    )
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        CONFIG.require_keys()
    except RuntimeError as exc:
        print(f"[config error] {exc}", file=sys.stderr)
        return 2

    print(
        f"Models: annotator={CONFIG.gemini_model} | auditor={CONFIG.openai_model}",
        flush=True,
    )
    print(f"Input:  {args.input}  ->  Output: {args.out_dir}", flush=True)

    writer = OutputWriter(args.out_dir)
    orchestrator = Orchestrator(writer=writer, resume=not args.no_resume)

    rows = stream_rows(
        args.input,
        text_field=args.text_field,
        id_field=args.id_field,
        fmt=args.format,
    )

    started = time.time()
    stats = orchestrator.run(rows, limit=args.limit, delay=args.delay)
    elapsed = time.time() - started

    print("\n=== Run complete ===")
    print(f"  processed : {stats.total}")
    print(f"  PASS->gold: {stats.passed}")
    print(f"  FAIL->queue: {stats.failed}")
    print(f"  pipeline errors: {stats.errored}")
    print(f"  skipped (resume): {stats.skipped}")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"\nOutputs written under: {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
