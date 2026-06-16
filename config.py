"""Environment-driven configuration for the labeling pipeline.

Models are selected per-role as ``"<provider>:<model>"`` specs so either stage
can use any provider and models can be swapped without code changes. Defaults
reflect the current setup: GPT annotates, Gemini judges.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

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

    # Role -> "<provider>:<model>". Swap models/providers here or via CLI flags.
    annotator_model: str = os.getenv("ANNOTATOR_MODEL", "openai:gpt-5.5")
    auditor_model: str = os.getenv("AUDITOR_MODEL", "gemini:gemini-3.5-flash")

    annotator_temperature: float = _get_float("ANNOTATOR_TEMPERATURE", 0.0)
    auditor_temperature: float = _get_float("AUDITOR_TEMPERATURE", 0.0)

    output_dir: str = os.getenv("OUTPUT_DIR", "output")

    def require_keys(self, model_specs: Iterable[str]) -> None:
        """Fail fast if an API key for an actually-used provider is missing.

        ``model_specs`` are the ``"provider:model"`` strings in play this run;
        only the providers they reference are required.
        """
        needed = {
            spec.split(":", 1)[0].strip().lower()
            for spec in model_specs
            if spec
        }
        missing = []
        if "gemini" in needed and not self.gemini_api_key:
            missing.append("GEMINI_API_KEY")
        if "openai" in needed and not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if missing:
            raise RuntimeError(
                "Missing required environment variable(s): "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill them in."
            )


CONFIG = Config()
