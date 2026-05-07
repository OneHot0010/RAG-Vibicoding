"""Base abstractions for ingestion transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.types import Chunk


class TransformError(RuntimeError):
    """Raised when a transform cannot be configured or executed."""


class BaseTransform(ABC):
    """Common interface for chunk transforms."""

    @abstractmethod
    def transform(self, chunks: list[Chunk], trace: Any | None = None) -> list[Chunk]:
        """Transform chunks while preserving the Chunk contract."""
