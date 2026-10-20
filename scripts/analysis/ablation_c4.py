"""Pipeline ablation C4: same-family audit (Gemini auditing Gemini).

Re-audits the cached Gemini annotations of the validation run with a Gemini
auditor, using the pipeline's own Auditor and prompt. Only the audit call is
new; the annotator is not re-run, so ỹ is identical to C1–C3.

Two subcommands:

  run    Call the auditor for every row with annotator output and append the
         verdicts to <out-dir>/audit_results.csv. Resumable: rows that already
         have a verdict are skipped; rows that errored (PIPELINE_ERROR) are
         retried on the next run.

  score  Offline. Report the flagged rows, compare them to the cross-family
         (GPT) audit, and compute post-human P/R/F1 against the ground truth.
         Escalated rows take the real reviewer annotation from
         human_reviewed.csv when one exists. Rows the cross-family audit never
         escalated have no reviewer annotation; they are written to
         <out-dir>/needs_review.csv in the review-queue format, and scored with
         the ground truth as a stand-in (a perfect reviewer) until a reviewed
         file is passed with --reviewed-csv.

Examples
--------
  python scripts/analysis/ablation_c4.py run --concurrency 8
  python scripts/analysis/ablation_c4.py score
  python scripts/analysis/ablation_c4.py score --reviewed-csv validation/general/c4_same_family/needs_review.csv
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

RUN_DIR = abl.RUN_DIR
OUT_DIR = RUN_DIR / "c4_same_family"
AUDIT_FILE = "audit_results.csv"
NEEDS_REVIEW_FILE = "needs_review.csv"

_AUDIT_HEADER = ["row_id", "status", "error_type", "auditor_reason"]
_REVIEW_HEADER = ["row_id", "raw_text", "tagged_text", "error_type", "auditor_reason", "human_annotation"]


def _load_audits(path: Path) -> dict[str, dict]:
    """Latest verdict per row_id (later lines override earlier retries)."""
    if not path.exists():
        return {}
    return {r["row_id"]: r for r in abl._read(path)}


# ── run ──────────────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> None:
    from pipeline.auditor import Auditor
    from pipeline.providers import build_provider
    from pipeline.schemas import AuditResult, AuditStatus, ErrorType

    CONFIG.require_keys([args.auditor_model])
    auditor = Auditor(provider=build_provider(args.auditor_model), temperature=args.temperature)

    truth = {r["row_id"]: r for r in abl._read(args.human_csv)}
    gemini = {r["row_id"]: r["tagged_text"] for r in abl._read(args.run_dir / "annotator_cache.csv")}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.out_dir / AUDIT_FILE
    if not audit_path.exists():
        with audit_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_AUDIT_HEADER)
    done = {rid for rid, r in _load_audits(audit_path).items() if r["error_type"] != "PIPELINE_ERROR"}

    # Rows without annotator output never reach the auditor in the pipeline.
    todo = [rid for rid in sorted(truth) if rid in gemini and rid not in done]
    print(f"Auditor: {args.auditor_model} (temperature={auditor.temperature})")
    print(f"Rows: {len(truth)}  annotated: {len(gemini)}  already audited: {len(done)}  to audit: {len(todo)}")

    sem = asyncio.Semaphore(args.concurrency)
    stats: Counter[str] = Counter()

    async def _one(rid: str) -> None:
        async with sem:
            try:
                res = await auditor.audit_async(truth[rid]["text"], gemini[rid])
            except Exception as exc:  # noqa: BLE001 - isolate per-row failures
                res = AuditResult(
                    status=AuditStatus.FAIL,
                    error_type=ErrorType.PIPELINE_ERROR,
                    auditor_reason=f"{type(exc).__name__}: {exc}",
                )
        with audit_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([rid, res.status.value, res.error_type.value, res.auditor_reason])
        stats[res.error_type.value if res.error_type == ErrorType.PIPELINE_ERROR else res.status.value] += 1
        n = sum(stats.values())
        if n % 25 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}  " + "  ".join(f"{k}={v}" for k, v in sorted(stats.items())), flush=True)

    await asyncio.gather(*(_one(rid) for rid in todo))
    if stats["PIPELINE_ERROR"]:
        print(f"\n{stats['PIPELINE_ERROR']} row(s) errored; rerun `run` to retry them.")
    print(f"Written: {audit_path}")


# ── score ────────────────────────────────────────────────────────────────────

def _score(args: argparse.Namespace) -> None:
    truth = {r["row_id"]: r for r in abl._read(args.human_csv)}
    gemini = {r["row_id"]: r["tagged_text"] for r in abl._read(args.run_dir / "annotator_cache.csv")}
    log = {r["row_id"]: r for r in abl._read(args.run_dir / "run_log.csv")}
    reviewed = {r["row_id"]: r["human_annotation"] for r in abl._read(args.run_dir / "human_reviewed.csv")}
    if args.reviewed_csv:
        extra = {r["row_id"]: r["human_annotation"] for r in abl._read(args.reviewed_csv)
                 if r.get("human_annotation", "").strip()}
        reviewed.update(extra)
    audits = _load_audits(args.out_dir / AUDIT_FILE)

    ids = sorted(truth)
    raw = {rid: truth[rid]["text"] for rid in ids}
    y = {rid: gemini.get(rid, "") for rid in ids}
    gold = {rid: abl._spans(truth[rid]["annotated_text"]) for rid in ids}

    no_output = {rid for rid in ids if rid not in gemini}
    unaudited = sorted(set(ids) - no_output - set(audits))
    errored = sorted(rid for rid, r in audits.items() if r["error_type"] == "PIPELINE_ERROR")
    if unaudited or errored:
        sys.exit(f"Audit incomplete: {len(unaudited)} row(s) not audited, {len(errored)} errored. "
                 "Run the `run` subcommand first.")

    gem_fail = {rid for rid, r in audits.items() if r["status"] == "FAIL"}
    gpt_fail = {rid for rid in ids if log[rid]["status"] == "FAIL"} - no_output
    parser_fail = {rid for rid in ids if abl._parser_rejects(y[rid], raw[rid])}

    # Hard errors (no annotator output) go to review in every condition, as in C3.
    conditions = {
        "C3 cross-family auditor-only": gpt_fail | no_output,
        "C4 same-family auditor-only": gem_fail | no_output,
        "C4 same-family + parser": gem_fail | no_output | parser_fail,
    }

    # Flagged-row summary and agreement with the cross-family audit.
    print(f"Rows: {len(ids)}   audited: {len(audits)}   no annotator output: {len(no_output)}\n")
    print(f"Gemini auditor FAIL : {len(gem_fail)} ({len(gem_fail)/len(ids):.1%})")
    print(f"GPT auditor FAIL    : {len(gpt_fail)} ({len(gpt_fail)/len(ids):.1%})")
    print(f"  flagged by both   : {len(gem_fail & gpt_fail)}")
    print(f"  Gemini only       : {len(gem_fail - gpt_fail)}")
    print(f"  GPT only          : {len(gpt_fail - gem_fail)}  (cross-family catches the same family missed)")
    print("\nGemini FAIL by error type:")
    for et, n in Counter(audits[rid]["error_type"] for rid in gem_fail).most_common():
        print(f"  {et:<22} {n}")

    # Rows that are truly wrong (raw Gemini spans differ from ground truth): auditor sensitivity.
    wrong = {rid for rid in set(ids) - no_output if abl._spans(y[rid]) != gold[rid]}
    print(f"\nRows where raw Gemini spans differ from ground truth: {len(wrong)}")
    for name, flagged in (("Gemini", gem_fail), ("GPT", gpt_fail)):
        caught = len(flagged & wrong)
        print(f"  {name:<6} auditor flags {caught}/{len(wrong)} of them; "
              f"{len(flagged - wrong)} flagged rows were already correct")

    # Flagged rows without a real reviewer annotation.
    all_escalated = set().union(*conditions.values())
    missing = sorted(all_escalated - set(reviewed))
    if missing:
        path = args.out_dir / NEEDS_REVIEW_FILE
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(_REVIEW_HEADER)
            for rid in missing:
                a = audits.get(rid, {})
                w.writerow([rid, raw[rid], y[rid], a.get("error_type", ""), a.get("auditor_reason", ""), ""])
        print(f"\n{len(missing)} escalated row(s) have no reviewer annotation; scored with the ground truth "
              f"as a stand-in.\nWritten for review: {path}")

    def _post_human(rid: str) -> set:
        return abl._spans(reviewed[rid]) if rid in reviewed else gold[rid]

    header = f"{'Condition':<30} {'Esc':>4} {'Rate':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>7} {'R':>7} {'F1':>7}"
    print("\n" + header)
    print("-" * len(header))
    for name, escalated in conditions.items():
        tp = fp = fn = 0
        for rid in ids:
            pred = _post_human(rid) if rid in escalated else abl._spans(y[rid])
            tp += len(pred & gold[rid])
            fp += len(pred - gold[rid])
            fn += len(gold[rid] - pred)
        p, r, f = abl._prf(tp, fp, fn)
        print(f"{name:<30} {len(escalated):>4} {len(escalated)/len(ids):>6.1%} "
              f"{tp:>4} {fp:>4} {fn:>4} {p:>7.4f} {r:>7.4f} {f:>7.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "score"):
        s = sub.add_parser(name)
        s.add_argument("--run-dir", type=Path, default=RUN_DIR)
        s.add_argument("--human-csv", type=Path, default=RUN_DIR / "sample_500_human.csv")
        s.add_argument("--out-dir", type=Path, default=OUT_DIR)
    run = sub.choices["run"]
    run.add_argument("--auditor-model", default=CONFIG.annotator_model,
                     help="Same-family auditor (default: the annotator model, $ANNOTATOR_MODEL)")
    run.add_argument("--temperature", type=float, default=None,
                     help="Auditor temperature (default: $AUDITOR_TEMPERATURE, as in the pipeline)")
    run.add_argument("--concurrency", type=int, default=8)
    sub.choices["score"].add_argument(
        "--reviewed-csv", type=Path, default=None,
        help="needs_review.csv with the human_annotation column filled in")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "run":
        asyncio.run(_run(args))
    else:
        _score(args)


if __name__ == "__main__":
    main()
