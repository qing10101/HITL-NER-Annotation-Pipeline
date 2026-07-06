"""Provider-agnostic LLM access layer.

Each pipeline stage talks to an ``LLMProvider`` rather than a specific SDK, so
models can be swapped freely via configuration. A model is named by a
``"<provider>:<model>"`` spec, e.g. ``"openai:gpt-5.4-mini"`` or
``"gemini:gemini-3.5-flash"``.

To add a new backend (e.g. Anthropic, a local server): subclass ``LLMProvider``,
implement ``generate_text`` + ``generate_structured``, and register it in
``_REGISTRY``. Nothing else in the pipeline needs to change.

SDK imports are lazy (inside each provider's ``__init__``) so you only need the
SDK for the provider(s) you actually use installed.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Type

from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
import logging

from config import CONFIG

logging.basicConfig(format="%(message)s", level=logging.WARNING)
_retry_logger = logging.getLogger("pipeline.retry")
_retry_logger.setLevel(logging.INFO)


class ContentBlocked(RuntimeError):
    """A model refused to produce output (safety/prohibited-content block).

    Non-retryable: the same input will be blocked every time, so retrying only
    wastes time and routes the row to human review more slowly.
    """


def _should_retry(exc: BaseException) -> bool:
    """Retry only errors that a later attempt could plausibly fix.

    NOT retried (fail fast -> row goes straight to the review queue):
      - content refusals (Gemini PROHIBITED_CONTENT / empty text) — ``ContentBlocked``
      - quota/credit exhaustion (HTTP 429 / RESOURCE_EXHAUSTED / insufficient_quota)
    Everything else (timeouts, 500/503/504, transport errors) is transient and
    still retried with backoff.
    """
    if isinstance(exc, ContentBlocked):
        return False
    msg = str(exc).lower()
    if "429" in msg or "resource_exhausted" in msg or "insufficient_quota" in msg:
        return False
    return True


# Shared retry policy: survives transient errors and rate-limit backoffs
# (3 attempts total = 1 initial + 2 retries; waits 4,8s). Content refusals and
# quota-exhaustion errors bypass retry entirely (see _should_retry).
_RETRY = retry(
    retry=retry_if_exception(_should_retry),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=70),
    before_sleep=before_sleep_log(_retry_logger, logging.INFO),
    reraise=True,
)


class LLMProvider(ABC):
    """Minimal capability surface the pipeline needs from any model backend."""

    provider_name: str = "base"

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def __str__(self) -> str:  # e.g. "openai:gpt-5.5"
        return f"{self.provider_name}:{self.model}"

    @abstractmethod
    def generate_text(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> str:
        """Free-form text generation (used by the annotator stage)."""

    @abstractmethod
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        schema: Type[BaseModel],
    ) -> Optional[BaseModel]:
        """Schema-constrained JSON generation (used by the auditor stage).

        Returns a validated instance of ``schema``, or ``None`` if the model
        produced nothing parseable.
        """

    async def generate_text_async(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> str:
        """Async wrapper — runs generate_text in the default thread-pool executor."""
        return await asyncio.to_thread(
            self.generate_text, system_prompt, user_prompt, temperature
        )

    async def generate_structured_async(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        schema: Type[BaseModel],
    ) -> Optional[BaseModel]:
        """Async wrapper — runs generate_structured in the default thread-pool executor."""
        return await asyncio.to_thread(
            self.generate_structured, system_prompt, user_prompt, temperature, schema
        )


def _is_temperature_unsupported(exc: Exception) -> bool:
    """True if an OpenAI 400 rejected a non-default ``temperature``.

    Reasoning-style models (e.g. gpt-5.5) only allow the default temperature.
    """
    msg = str(exc).lower()
    return "temperature" in msg and (
        "does not support" in msg
        or "unsupported" in msg
        or "only the default" in msg
    )


class OpenAIProvider(LLMProvider):
    provider_name = "openai"

    def __init__(self, model: str, api_key: str):
        super().__init__(model, api_key)
        import httpx
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key,
            timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=10.0),
        )
        # Some models reject a non-default temperature; flip off after the first
        # 400 so the rest of the run omits the param entirely.
        self._send_temperature = True

    def _with_temp_fallback(self, make_call, temperature):
        """Call ``make_call(temp_kwargs)``; on a temperature-unsupported 400,
        retry once without the temperature param and remember the decision."""
        temp_kwargs = {"temperature": temperature} if self._send_temperature else {}
        try:
            return make_call(temp_kwargs)
        except Exception as exc:  # noqa: BLE001
            if self._send_temperature and _is_temperature_unsupported(exc):
                self._send_temperature = False
                return make_call({})
            raise

    @_RETRY
    def generate_text(self, system_prompt, user_prompt, temperature):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        completion = self._with_temp_fallback(
            lambda tk: self._client.chat.completions.create(
                model=self.model, messages=messages, **tk
            ),
            temperature,
        )
        content = completion.choices[0].message.content
        if content is None:
            raise RuntimeError("OpenAI returned empty content.")
        return content

    @_RETRY
    def generate_structured(self, system_prompt, user_prompt, temperature, schema):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        completion = self._with_temp_fallback(
            lambda tk: self._client.beta.chat.completions.parse(
                model=self.model, messages=messages, response_format=schema, **tk
            ),
            temperature,
        )
        return completion.choices[0].message.parsed


class GeminiProvider(LLMProvider):
    provider_name = "gemini"

    def __init__(self, model: str, api_key: str):
        super().__init__(model, api_key)
        from google import genai
        from google.genai import types

        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=120_000),  # milliseconds
        )

    def _config(self, system_prompt: str, temperature: float, **extra):
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=system_prompt, temperature=temperature, **extra
        )

    @_RETRY
    def generate_text(self, system_prompt, user_prompt, temperature):
        resp = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=self._config(system_prompt, temperature),
        )
        if resp.text is None:
            reason = getattr(
                getattr(resp, "prompt_feedback", None), "block_reason", None
            )
            raise ContentBlocked(
                f"Gemini returned no text (block_reason={reason})."
            )
        return resp.text

    @_RETRY
    def generate_structured(self, system_prompt, user_prompt, temperature, schema):
        resp = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=self._config(
                system_prompt,
                temperature,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        # google-genai populates .parsed with a schema instance when
        # response_schema is a pydantic model.
        return resp.parsed


_REGISTRY = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
}


def known_providers() -> list[str]:
    return sorted(_REGISTRY)


def parse_spec(spec: str) -> tuple[str, str]:
    """Split ``"provider:model"`` into ``("provider", "model")`` with validation."""
    provider, sep, model = spec.partition(":")
    provider = provider.strip().lower()
    model = model.strip()
    if not sep or not provider or not model:
        raise ValueError(
            f"Invalid model spec {spec!r}; expected '<provider>:<model>' "
            f"(e.g. 'openai:gpt-5.5'). Known providers: {known_providers()}"
        )
    if provider not in _REGISTRY:
        raise ValueError(
            f"Unknown provider {provider!r} in {spec!r}; known: {known_providers()}"
        )
    return provider, model


def _api_key_for(provider: str) -> str:
    return {
        "openai": CONFIG.openai_api_key,
        "gemini": CONFIG.gemini_api_key,
    }.get(provider, "")


def build_provider(spec: str) -> LLMProvider:
    """Construct a provider instance from a ``"provider:model"`` spec."""
    provider, model = parse_spec(spec)
    return _REGISTRY[provider](model=model, api_key=_api_key_for(provider))
