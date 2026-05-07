"""Base abstractions for vector storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class VectorRecord:
    """A single record to persist in a vector store."""

    id: str
    vector: list[float]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorQueryResult:
    """A single vector query hit returned by a vector store."""

    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseVectorStore(ABC):
    """Common interface for vector database backends."""

    @abstractmethod
    def upsert(self, records: list[VectorRecord], trace: Any | None = None) -> None:
        """Insert or replace records by id."""

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[VectorQueryResult]:
        """Return the top matching records for a query vector."""
