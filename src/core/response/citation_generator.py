"""Citation generation for retrieval results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.types import RetrievalResult


@dataclass(frozen=True)
class Citation:
    """Structured citation attached to an MCP tool response."""

    index: int
    source: str
    page: int | str | None
    chunk_id: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize citation fields for structuredContent."""
        return {
            "index": self.index,
            "source": self.source,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "score": self.score,
        }


class CitationGenerator:
    """Build stable citations from retrieval results."""

    def generate(self, retrieval_results: list[RetrievalResult]) -> list[Citation]:
        """Return one citation per retrieval result."""
        _validate_results(retrieval_results)
        citations: list[Citation] = []
        for index, result in enumerate(retrieval_results, start=1):
            citations.append(
                Citation(
                    index=index,
                    source=str(result.metadata.get("source_path") or result.metadata.get("source") or "unknown"),
                    page=result.metadata.get("page") or result.metadata.get("page_num"),
                    chunk_id=result.chunk_id,
                    score=float(result.score),
                )
            )
        return citations


def _validate_results(retrieval_results: list[RetrievalResult]) -> None:
    if not isinstance(retrieval_results, list):
        raise ValueError("retrieval_results must be a list")
    for result in retrieval_results:
        if not isinstance(result, RetrievalResult):
            raise ValueError("retrieval_results must contain RetrievalResult objects")
