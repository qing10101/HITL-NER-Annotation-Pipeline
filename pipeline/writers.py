"""Step 4 — CSV output writers (replaces the Postgres target DB).

Four append-mode CSV files, all keyed by ``row_id`` so every output row maps
back to its source row number:

  - gold_standard.csv    : one row per PASS record (committed gold data)
  - gold_spans.csv       : exploded, one row per labeled entity span
  - review_queue.csv     : one row per FAIL record (Human-in-the-Loop queue)
  - run_log.csv          : one row per processed record (audit trail)
  - annotator_cache.csv  : tagged_text saved immediately after annotation,
                           before the auditor runs — allows auditor-only retries
                           without re-spending the annotator API call.

Each writer creates its header on first open and supports resumability by
exposing the set of already-processed row_ids.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Set

from .schemas import RowResult, Span

GOLD_FILE = "gold_standard.csv"
SPANS_FILE = "gold_spans.csv"
REVIEW_FILE = "review_queue.csv"
LOG_FILE = "run_log.csv"
ANNOTATOR_CACHE_FILE = "annotator_cache.csv"

_GOLD_HEADER = ["row_id", "raw_text", "tagged_text", "num_entities", "entities_json"]
_SPANS_HEADER = ["row_id", "entity_index", "label", "text", "start", "end"]
_REVIEW_HEADER = ["row_id", "raw_text", "tagged_text", "error_type", "auditor_reason"]
_CACHE_HEADER = ["row_id", "tagged_text"]
_LOG_HEADER = ["row_id", "status", "error_type", "num_entities", "note"]


def _spans_to_json(spans: List[Span]) -> str:
    return json.dumps([s.model_dump() for s in spans], ensure_ascii=False)


class OutputWriter:
    """Manages the four CSV sinks under ``output_dir``."""

    def __init__(self, output_dir: str | Path):
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._gold = self.dir / GOLD_FILE
        self._spans = self.dir / SPANS_FILE
        self._review = self.dir / REVIEW_FILE
        self._log = self.dir / LOG_FILE
        self._cache = self.dir / ANNOTATOR_CACHE_FILE
        self._ensure_headers()

    def _ensure_headers(self) -> None:
        for path, header in (
            (self._gold, _GOLD_HEADER),
            (self._spans, _SPANS_HEADER),
            (self._review, _REVIEW_HEADER),
            (self._log, _LOG_HEADER),
            (self._cache, _CACHE_HEADER),
        ):
            if not path.exists():
                with path.open("w", encoding="utf-8", newline="") as fh:
                    csv.writer(fh).writerow(header)

    def load_annotator_cache(self) -> dict[str, str]:
        """Return {row_id: tagged_text} for every row already annotated."""
        if not self._cache.exists():
            return {}
        cache: dict[str, str] = {}
        with self._cache.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("row_id"):
                    cache[row["row_id"]] = row["tagged_text"]
        return cache

    def save_annotator_result(self, row_id: str, tagged_text: str) -> None:
        """Persist a single annotator result immediately after it is produced."""
        self._append(self._cache, [row_id, tagged_text])

    def processed_ids(self) -> Set[str]:
        """Row ids already present in run_log.csv (for resumable re-runs)."""
        if not self._log.exists():
            return set()
        ids: Set[str] = set()
        with self._log.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rid = row.get("row_id")
                if rid:
                    ids.add(rid)
        return ids

    @staticmethod
    def _append(path: Path, row: list) -> None:
        with path.open("a", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerow(row)

    def write(self, result: RowResult) -> None:
        """Persist one fully-processed row to the appropriate sinks + the log."""
        status = result.audit.status.value if result.audit else "FAIL"
        error_type = result.audit.error_type.value if result.audit else "PIPELINE_ERROR"

        if result.committed:
            self._append(
                self._gold,
                [
                    result.row_id,
                    result.raw_text,
                    result.tagged_text,
                    len(result.spans),
                    _spans_to_json(result.spans),
                ],
            )
            for idx, span in enumerate(result.spans):
                self._append(
                    self._spans,
                    [result.row_id, idx, span.label, span.text, span.start, span.end],
                )
        else:
            self._append(
                self._review,
                [
                    result.row_id,
                    result.raw_text,
                    result.tagged_text,
                    error_type,
                    result.audit.auditor_reason if result.audit else result.note,
                ],
            )

        self._append(
            self._log,
            [result.row_id, status, error_type, len(result.spans), result.note],
        )
