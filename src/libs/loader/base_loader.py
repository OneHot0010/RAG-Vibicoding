"""Base abstractions for document loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from core.types import Document


class LoaderError(RuntimeError):
    """Raised when a loader cannot parse an input document."""


class BaseLoader(ABC):
    """Common interface for document loaders."""

    @abstractmethod
    def load(self, path: str | Path) -> Document:
        """Load a file into the normalized Document contract."""
