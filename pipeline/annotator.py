"""Step 2 — Context-aware in-line tagging engine (gemini-3.5-flash).

Forwards the raw review to Gemini, which returns the text verbatim with inline
XML tags injected around target entities. The model never emits coordinates.
"""
from __future__ import annotations

import re

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from config import CONFIG
from .prompts import ANNOTATOR_SYSTEM_PROMPT, annotator_user_prompt

_FENCE_OPEN = re.compile(r"^```[a-zA-Z]*\n?")
_FENCE_CLOSE = re.compile(r"\n?```$")
_LABEL_PREFIX = re.compile(r'^\s*Review Text:\s*', re.IGNORECASE)


def _clean_output(text: str) -> str:
    """Strip any scaffolding the model may echo back around the rewritten text.

    Defensive only — the prompt instructs the model to emit the bare review.
    Removes markdown code fences, a leading ``Review Text:`` label, and a single
    layer of wrapping triple quotes.
    """
    t = text.strip()
    if t.startswith("```"):
        t = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", t)).strip()
    t = _LABEL_PREFIX.sub("", t)
    if len(t) >= 6 and t.startswith('"""') and t.endswith('"""'):
        t = t[3:-3]
    return t.strip("\n")


class Annotator:
    """Stage-1 inline tagging engine."""

    def __init__(self, model: str | None = None, temperature: float | None = None):
        self.model = model or CONFIG.gemini_model
        self.temperature = (
            CONFIG.annotator_temperature if temperature is None else temperature
        )
        self._client = genai.Client(api_key=CONFIG.gemini_api_key)

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=4, max=70),
        reraise=True,
    )
    def tag(self, raw_text: str) -> str:
        """Return ``tagged_text`` for a single normalized review."""
        response = self._client.models.generate_content(
            model=self.model,
            contents=annotator_user_prompt(raw_text),
            config=types.GenerateContentConfig(
                system_instruction=ANNOTATOR_SYSTEM_PROMPT,
                temperature=self.temperature,
            ),
        )
        text = response.text
        if text is None:
            raise RuntimeError("Annotator returned no text (possibly blocked/empty).")
        return _clean_output(text)
