"""Base abstractions for text splitting strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSplitter(ABC):
    """Common interface for document text splitters."""

    @abstractmethod
    def split_text(self, text: str, trace: Any | None = None) -> list[str]:
        """Split text into ordered chunks."""
