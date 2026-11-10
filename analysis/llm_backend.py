"""Backend interfaces for structured language-model generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class LLMGenerationResult:
    """Raw generated text and provider metadata for one model request."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMBackend(ABC):
    """Interface implemented by local structured-generation backends."""

    name = "llm"
    is_real = False

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        *,
        timeout_seconds: float,
        options: Mapping[str, Any],
    ) -> LLMGenerationResult:
        """Generate JSON-formatted text and return its provider metadata."""
