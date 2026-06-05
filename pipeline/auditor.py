"""Step 3 — Independent cross-family guard auditor (gpt-5.4-mini).

Switches model ecosystems to break single-model confirmation bias. Receives the
pristine RAW_TEXT and the ANNOTATED_TEXT side by side and returns a strict JSON
verdict via structured outputs (Pydantic schema constraint).
"""
from __future__ import annotations

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import CONFIG
from .prompts import AUDITOR_SYSTEM_PROMPT, auditor_user_prompt
from .schemas import AuditResult, AuditStatus, ErrorType


class Auditor:
    """Stage-2 structural + semantic verification engine."""

    def __init__(self, model: str | None = None, temperature: float | None = None):
        self.model = model or CONFIG.openai_model
        self.temperature = (
            CONFIG.auditor_temperature if temperature is None else temperature
        )
        self._client = OpenAI(api_key=CONFIG.openai_api_key)

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=4, max=70),
        reraise=True,
    )
    def audit(self, raw_text: str, tagged_text: str) -> AuditResult:
        """Return the structured PASS/FAIL verdict for one row."""
        completion = self._client.beta.chat.completions.parse(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": AUDITOR_SYSTEM_PROMPT},
                {"role": "user", "content": auditor_user_prompt(raw_text, tagged_text)},
            ],
            response_format=AuditResult,
        )
        result = completion.choices[0].message.parsed
        if result is None:
            # Model refused / could not produce a schema-valid object.
            return AuditResult(
                status=AuditStatus.FAIL,
                error_type=ErrorType.PIPELINE_ERROR,
                auditor_reason="Auditor returned no parseable structured output.",
            )
        return result
