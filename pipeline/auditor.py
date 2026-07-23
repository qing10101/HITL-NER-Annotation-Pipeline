"""Step 3 — Independent cross-family guard auditor (the judge).

Receives the pristine RAW_TEXT and the ANNOTATED_TEXT side by side and returns a
strict JSON verdict via schema-constrained structured output. The backing model
is provider-agnostic — configure it via ``AUDITOR_MODEL`` (e.g.
``gemini:gemini-3.5-flash``). "Cross-family" auditing simply means choosing a
different provider than the annotator, which breaks single-model confirmation
bias.
"""
from __future__ import annotations

from .config import CONFIG
from .prompts import AUDITOR_SYSTEM_PROMPT, auditor_user_prompt
from .providers import LLMProvider, build_provider
from .schemas import AuditResult, AuditStatus, ErrorType


class Auditor:
    """Stage-2 structural + semantic verification engine (model-agnostic)."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        temperature: float | None = None,
    ):
        self.provider = provider or build_provider(CONFIG.auditor_model)
        self.temperature = (
            CONFIG.auditor_temperature if temperature is None else temperature
        )

    def audit(self, raw_text: str, tagged_text: str) -> AuditResult:
        """Return the structured PASS/FAIL verdict for one row."""
        result = self.provider.generate_structured(
            AUDITOR_SYSTEM_PROMPT,
            auditor_user_prompt(raw_text, tagged_text),
            self.temperature,
            AuditResult,
        )
        if not isinstance(result, AuditResult):
            # Model refused / produced nothing schema-valid.
            return AuditResult(
                status=AuditStatus.FAIL,
                error_type=ErrorType.PIPELINE_ERROR,
                auditor_reason="Auditor returned no parseable structured output.",
            )
        return result

    async def audit_async(self, raw_text: str, tagged_text: str) -> AuditResult:
        """Async version of audit() for concurrent batch processing."""
        result = await self.provider.generate_structured_async(
            AUDITOR_SYSTEM_PROMPT,
            auditor_user_prompt(raw_text, tagged_text),
            self.temperature,
            AuditResult,
        )
        if not isinstance(result, AuditResult):
            return AuditResult(
                status=AuditStatus.FAIL,
                error_type=ErrorType.PIPELINE_ERROR,
                auditor_reason="Auditor returned no parseable structured output.",
            )
        return result
