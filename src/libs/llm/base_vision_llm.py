"""Base abstractions for multimodal vision LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChatResponse:
    """Normalized response returned by vision LLM providers."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseVisionLLM(ABC):
    """Common interface for LLM backends that accept text plus one image."""

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any | None = None,
    ) -> ChatResponse:
        """Return a text response for a prompt and image input."""
