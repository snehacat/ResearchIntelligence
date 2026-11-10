"""Select a language-model backend from application configuration."""

import os
from typing import Optional

from analysis.llm_backend import LLMBackend


DEFAULT_BACKEND = "groq"
DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-20b",
}


def configured_backend_name() -> str:
    """Return the configured backend name, defaulting to Groq."""
    return os.environ.get(
        "RESEARCHINTELLIGENCE_LLM_BACKEND", DEFAULT_BACKEND
    ).strip().casefold()


def configured_model_name(backend_name: str) -> Optional[str]:
    """Return an explicit model override or the provider's known default."""
    model_name = os.environ.get("RESEARCHINTELLIGENCE_LLM_MODEL")
    if model_name and model_name.strip():
        return model_name.strip()
    if backend_name == "gemini":
        return None
    return DEFAULT_MODELS.get(backend_name)


def create_llm_backend(
    backend_name: str,
    model_name: str,
) -> LLMBackend:
    """Construct one of the supported backends without provider logic in callers."""
    if backend_name == "groq":
        from analysis.groq_backend import GroqBackend

        return GroqBackend(model_name=model_name)
    if backend_name == "gemini":
        from analysis.gemini_backend import GeminiBackend

        return GeminiBackend(model_name=model_name)
    raise ValueError(
        "RESEARCHINTELLIGENCE_LLM_BACKEND must be one of: groq, gemini"
    )
