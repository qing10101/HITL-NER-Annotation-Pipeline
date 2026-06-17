"""Typed data contracts for the pipeline."""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class AuditStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class ErrorType(str, Enum):
    """Error taxonomy fixed by the Stage-2 auditor prompt.

    Names mirror the auditor system prompt's FAIL conditions exactly, because
    the OpenAI structured-output schema is generated from this enum — any drift
    between the prompt's category names and these members would make the model's
    intended verdict unrepresentable.

    ``PIPELINE_ERROR`` is an internal addition for rows that fail with an
    exception (e.g. a transport error or malformed tag structure) rather than a
    semantic verdict; it is never emitted by the model.
    """

    NONE = "NONE"
    RAW_TEXT_MUTATION = "RAW_TEXT_MUTATION"
    NON_HUMAN_TAGGING = "NON_HUMAN_TAGGING"
    UNANCHORED_TAGGING = "UNANCHORED_TAGGING"
    OMITTED_VALID_TAG = "OMITTED_VALID_TAG"
    MISALLOCATED_LABEL = "MISALLOCATED_LABEL"
    INVALID_SPAN_BOUNDARY = "INVALID_SPAN_BOUNDARY"
    OUT_OF_SCOPE_TAG = "OUT_OF_SCOPE_TAG"
    PIPELINE_ERROR = "PIPELINE_ERROR"


class AuditResult(BaseModel):
    """Structured output the Stage-2 auditor (gpt-5.5) must return."""

    status: AuditStatus
    error_type: ErrorType = ErrorType.NONE
    auditor_reason: str = ""


class Span(BaseModel):
    """A single labeled entity with 0-based, half-open char offsets into raw_text."""

    label: str
    text: str
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)


class RowResult(BaseModel):
    """Full per-row outcome carried through the orchestrator."""

    row_id: str
    raw_text: str
    tagged_text: str = ""
    audit: Optional[AuditResult] = None
    spans: List[Span] = Field(default_factory=list)
    committed: bool = False  # True => gold-standard; False => review queue
    note: str = ""
