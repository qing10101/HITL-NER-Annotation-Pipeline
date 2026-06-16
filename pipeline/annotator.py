"""Step 2 — Context-aware in-line tagging engine (the annotator).

Sends the raw review to the configured annotator model, which returns the text
verbatim with inline XML tags injected around target entities. The model never
emits coordinates. The backing model is provider-agnostic — configure it via
``ANNOTATOR_MODEL`` (e.g. ``openai:gpt-5.5``); see providers.py.
"""
from __future__ import annotations

import re

from config import CONFIG
from .prompts import ANNOTATOR_SYSTEM_PROMPT, annotator_user_prompt
from .providers import LLMProvider, build_provider

_FENCE_OPEN = re.compile(r"^```[a-zA-Z]*\n?")
_FENCE_CLOSE = re.compile(r"\n?```$")
_LABEL_PREFIX = re.compile(r'^\s*Review Text:\s*', re.IGNORECASE)


def _clean_output(text: str) -> str:
    """Strip any scaffolding the model may echo back around the rewritten text.

    Defensive only — the prompt instructs the model to emit the bare review.
    Removes markdown code fences, a leading ``Review Text:`` label, and a single
    layer of wrapping triple quotes. Does NOT touch XML tags.
    """
    t = text.strip()
    if t.startswith("```"):
        t = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", t)).strip()
    t = _LABEL_PREFIX.sub("", t)
    if len(t) >= 6 and t.startswith('"""') and t.endswith('"""'):
        t = t[3:-3]
    return t.strip("\n")


class Annotator:
    """Stage-1 inline tagging engine (model-agnostic)."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        temperature: float | None = None,
    ):
        self.provider = provider or build_provider(CONFIG.annotator_model)
        self.temperature = (
            CONFIG.annotator_temperature if temperature is None else temperature
        )

    def tag(self, raw_text: str) -> str:
        """Return ``tagged_text`` for a single normalized review."""
        text = self.provider.generate_text(
            ANNOTATOR_SYSTEM_PROMPT,
            annotator_user_prompt(raw_text),
            self.temperature,
        )
        return _clean_output(text)
