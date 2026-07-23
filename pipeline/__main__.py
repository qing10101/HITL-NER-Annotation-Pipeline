"""CLI entrypoint for the Cascading Multi-Agent Inline Tagging & Audit pipeline.

Run as a module from the repo root (this is a package __main__, not a standalone
script — relative imports below require it):

Examples
--------
  python -m pipeline --input data/sample_reviews.jsonl
  python -m pipeline --input data/reviews.csv --text-field review --limit 500
  python -m pipeline --input data/reviews.jsonl --out-dir output/run1 --no-resume
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time

from .annotator import Annotator
from .auditor import Auditor
from .config import CONFIG
from .ingestion import count_rows, stream_rows
from .orchestrator import Orchestrator
from .providers import build_provider
from .writers import OutputWriter


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
    p.add_argument(
        "--annotator-model",
        default=None,
        help='Override annotator model, "<provider>:<model>" (default: $ANNOTATOR_MODEL)',
    )
    p.add_argument(
        "--auditor-model",
        default=None,
        help='Override auditor/judge model, "<provider>:<model>" (default: $AUDITOR_MODEL)',
    )
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="0-based input row position to start at (rows before it are skipped entirely)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Window length: process input rows [start, start+limit). "
        "Rerunning the same window skips rows already in run_log.csv (resume).",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between rows (throttle for rate-limited tiers)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Max rows processed concurrently (default: 8; use 1 to match old sequential behaviour)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocess rows already present in run_log.csv",
    )
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    ann_spec = args.annotator_model or CONFIG.annotator_model
    aud_spec = args.auditor_model or CONFIG.auditor_model

    try:
        CONFIG.require_keys([ann_spec, aud_spec])
        annotator = Annotator(provider=build_provider(ann_spec))
        auditor = Auditor(provider=build_provider(aud_spec))
    except (RuntimeError, ValueError) as exc:
        print(f"[config error] {exc}", file=sys.stderr)
        return 2

    print(f"Models: annotator={ann_spec} | auditor={aud_spec}", flush=True)
    print(f"Input:  {args.input}  ->  Output: {args.out_dir}", flush=True)

    writer = OutputWriter(args.out_dir)
    orchestrator = Orchestrator(
        writer=writer,
        annotator=annotator,
        auditor=auditor,
        resume=not args.no_resume,
    )

    rows = stream_rows(
        args.input,
        text_field=args.text_field,
        id_field=args.id_field,
        fmt=args.format,
    )

    # Pre-count for an accurate progress-bar total (best-effort). The total is
    # the window size: rows in [start, start+limit).
    try:
        total = max(0, count_rows(args.input, fmt=args.format) - args.start)
        if args.limit is not None:
            total = min(total, args.limit)
    except Exception:
        total = args.limit  # may be None -> indeterminate bar

    started = time.time()
    stats = asyncio.run(
        orchestrator.run_async(
            rows,
            start=args.start,
            limit=args.limit,
            concurrency=args.concurrency,
            delay=args.delay,
            total=total,
        )
    )
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
