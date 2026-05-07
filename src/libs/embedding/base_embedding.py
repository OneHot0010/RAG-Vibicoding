"""Base abstractions for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEmbedding(ABC):
    """Common interface for all embedding backends."""

    @abstractmethod
    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        """Return one embedding vector per input text."""
