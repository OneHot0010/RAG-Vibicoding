"""Base abstractions for reranking providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RerankCandidate:
    """A candidate item to rerank."""

    id: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankResult:
    """A reranked candidate with a final score and rank."""

    id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseReranker(ABC):
    """Common interface for all reranking backends."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Any | None = None,
    ) -> list[RerankResult]:
        """Return candidates sorted by relevance to the query."""


class NoneReranker(BaseReranker):
    """Reranker that preserves the original candidate order."""

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Any | None = None,
    ) -> list[RerankResult]:
        return [
            RerankResult(
                id=candidate.id,
                text=candidate.text,
                score=candidate.score,
                rank=index + 1,
                metadata=candidate.metadata,
            )
            for index, candidate in enumerate(candidates)
        ]
