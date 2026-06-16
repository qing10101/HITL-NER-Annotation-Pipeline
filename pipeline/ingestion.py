"""Step 1 — Input ingestion and dataset streaming.

Streams unstructured review rows one at a time (generator-based, so the 20,000+
row target never loads into memory at once) and applies the whitespace
normalization wrapper that establishes a pristine string baseline for
downstream spatial alignment.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterator, Tuple

# Standardized regex wrapper: strip leading/trailing whitespace only. Internal
# characters are preserved untouched (Character Preservation Mandate).
_LEADING = re.compile(r"^\s+")
_TRAILING = re.compile(r"\s+$")


def normalize(text: str) -> str:
    """Trim leading/trailing whitespace anomalies without touching inner chars."""
    return _TRAILING.sub("", _LEADING.sub("", text))


def _detect_format(path: Path, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "jsonl"  # treat .json as line-delimited; fall back below if needed
    return "csv"


def stream_rows(
    path: str | Path,
    text_field: str = "text",
    id_field: str = "id",
    fmt: str = "auto",
) -> Iterator[Tuple[str, str]]:
    """Yield ``(row_id, normalized_text)`` pairs one at a time.

    Accepts JSONL (one JSON object per line) or CSV with a text column. If a
    row has no id field, a stable zero-padded index (``row_00001``) is used so
    every output CSV can be keyed back to its source row number.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    resolved = _detect_format(path, fmt)

    if resolved == "jsonl":
        yield from _stream_jsonl(path, text_field, id_field)
    else:
        yield from _stream_csv(path, text_field, id_field)


def count_rows(path: str | Path, fmt: str = "auto") -> int:
    """Cheaply count records in the input (for a progress-bar total).

    Reads the file once without normalization; for JSONL counts non-empty
    lines, for CSV counts data rows (header excluded).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if _detect_format(path, fmt) == "jsonl":
        with path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    with path.open("r", encoding="utf-8", newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def _make_id(raw_id, index: int) -> str:
    if raw_id is None or str(raw_id).strip() == "":
        return f"row_{index + 1:05d}"
    return str(raw_id)


def _stream_jsonl(path: Path, text_field: str, id_field: str) -> Iterator[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        index = 0
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(text_field)
            if text is None:
                raise KeyError(
                    f"JSONL row {index + 1} has no '{text_field}' field; keys: {list(obj)}"
                )
            yield _make_id(obj.get(id_field), index), normalize(str(text))
            index += 1


def _stream_csv(path: Path, text_field: str, id_field: str) -> Iterator[Tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or text_field not in reader.fieldnames:
            raise KeyError(
                f"CSV must contain a '{text_field}' column; found: {reader.fieldnames}"
            )
        for index, row in enumerate(reader):
            text = row.get(text_field) or ""
            yield _make_id(row.get(id_field), index), normalize(text)
