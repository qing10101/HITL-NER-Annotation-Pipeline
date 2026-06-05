"""Environment-driven configuration for the labeling pipeline.

Model IDs default to the values named in the proposal document
(``gemini-3.5-flash`` / ``gpt-5.4-mini``) and can be overridden via ``.env``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

    annotator_temperature: float = _get_float("ANNOTATOR_TEMPERATURE", 0.0)
    auditor_temperature: float = _get_float("AUDITOR_TEMPERATURE", 0.0)

    output_dir: str = os.getenv("OUTPUT_DIR", "output")

    def require_keys(self) -> None:
        """Fail fast with a clear message if API keys are missing."""
        missing = []
        if not self.gemini_api_key:
            missing.append("GEMINI_API_KEY")
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if missing:
            raise RuntimeError(
                "Missing required environment variable(s): "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill them in."
            )


CONFIG = Config()
