"""The orchestrator — drives the per-row decision fork across the full stream.

Per row:
  1. annotate (gemini-3.5-flash)  -> tagged_text
  2. audit    (gpt-5.4-mini)      -> AuditResult
  3. PASS -> deterministic parse + invariant check -> gold output
     FAIL -> review queue
Any hard exception (transport, malformed tags) routes the row to review with
error_type=PIPELINE_ERROR so a single bad row never aborts the batch.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from .annotator import Annotator
from .auditor import Auditor
from .parser import TagParseError, parse_and_verify
from .schemas import AuditResult, AuditStatus, ErrorType, RowResult
from .writers import OutputWriter


@dataclass
class RunStats:
    total: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    skipped: int = 0


class Orchestrator:
    def __init__(
        self,
        writer: OutputWriter,
        annotator: Optional[Annotator] = None,
        auditor: Optional[Auditor] = None,
        resume: bool = True,
    ):
        self.writer = writer
        self.annotator = annotator or Annotator()
        self.auditor = auditor or Auditor()
        self._processed = writer.processed_ids() if resume else set()

    def process_row(self, row_id: str, raw_text: str) -> RowResult:
        result = RowResult(row_id=row_id, raw_text=raw_text)
        try:
            # Step 2: inline tagging.
            result.tagged_text = self.annotator.tag(raw_text)
            # Step 3: cross-family audit.
            audit = self.auditor.audit(raw_text, result.tagged_text)
            result.audit = audit

            if audit.status == AuditStatus.PASS:
                # Step 4A: deterministic offsets + invariant backstop.
                try:
                    result.spans = parse_and_verify(result.tagged_text, raw_text)
                    result.committed = True
                    result.note = "committed"
                except TagParseError as exc:
                    # Auditor passed it but the structure/character invariant
                    # failed -> override to FAIL and send to humans.
                    result.committed = False
                    result.audit = AuditResult(
                        status=AuditStatus.FAIL,
                        error_type=ErrorType.PIPELINE_ERROR,
                        auditor_reason=f"Parser rejected PASS row: {exc}",
                    )
                    result.note = "pass_overridden_by_parser"
            else:
                # Step 4B: human review queue.
                result.committed = False
                result.note = "audit_fail"

        except Exception as exc:  # noqa: BLE001 - isolate per-row failures
            result.committed = False
            result.audit = AuditResult(
                status=AuditStatus.FAIL,
                error_type=ErrorType.PIPELINE_ERROR,
                auditor_reason=f"{type(exc).__name__}: {exc}",
            )
            result.note = "pipeline_error"

        return result

    def run(
        self,
        rows: Iterable[Tuple[str, str]],
        limit: Optional[int] = None,
        progress_every: int = 25,
        delay: float = 0.0,
    ) -> RunStats:
        stats = RunStats()
        processed_this_run = 0
        for row_id, raw_text in rows:
            if limit is not None and stats.total >= limit:
                break
            if row_id in self._processed:
                stats.skipped += 1
                continue

            # Pace requests to respect provider rate limits (e.g. free tier).
            if delay > 0 and processed_this_run > 0:
                time.sleep(delay)
            processed_this_run += 1

            stats.total += 1
            result = self.process_row(row_id, raw_text)
            self.writer.write(result)
            self._processed.add(row_id)

            if result.committed:
                stats.passed += 1
            elif result.audit and result.audit.error_type == ErrorType.PIPELINE_ERROR:
                stats.errored += 1
            else:
                stats.failed += 1

            if stats.total % progress_every == 0:
                print(
                    f"  processed={stats.total} pass={stats.passed} "
                    f"fail={stats.failed} err={stats.errored}",
                    flush=True,
                )

        return stats
