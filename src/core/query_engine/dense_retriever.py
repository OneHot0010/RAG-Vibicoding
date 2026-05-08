"""Dense semantic retrieval backed by embeddings and a vector store."""

from __future__ import annotations

from typing import Any, Mapping

from core.settings import Settings
from core.types import RetrievalResult
from libs.embedding import BaseEmbedding, EmbeddingFactory
from libs.vector_store import BaseVectorStore, VectorQueryResult, VectorStoreFactory


class DenseRetrieverError(ValueError):
    """Raised when dense retrieval input or backend output is invalid."""


class DenseRetriever:
    """Embed a query and retrieve nearest chunks from the configured vector store."""

    def __init__(
        self,
        settings: Settings,
        embedding_client: BaseEmbedding | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_client = embedding_client or EmbeddingFactory.create(settings)
        self.vector_store = vector_store or VectorStoreFactory.create(settings)

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        """Return dense retrieval results for a query."""
        normalized_query = _normalize_query(query)
        if top_k <= 0:
            raise DenseRetrieverError("top_k must be greater than 0")
        normalized_filters = _normalize_filters(filters)

        vectors = self.embedding_client.embed([normalized_query], trace=trace)
        query_vector = _single_query_vector(vectors)
        hits = self.vector_store.query(query_vector, top_k=top_k, filters=normalized_filters, trace=trace)
        results = [_to_retrieval_result(hit) for hit in hits]
        _record_trace(
            trace,
            "dense_retriever.retrieve",
            {"top_k": top_k, "result_count": len(results), "filters": normalized_filters},
        )
        return results


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise DenseRetrieverError("query must be a non-empty string")
    return query.strip()


def _normalize_filters(filters: Mapping[str, Any] | None) -> dict[str, Any]:
    if filters is None:
        return {}
    if not isinstance(filters, Mapping):
        raise DenseRetrieverError("filters must be a mapping")
    return dict(filters)


def _single_query_vector(vectors: list[list[float]]) -> list[float]:
    if not isinstance(vectors, list) or len(vectors) != 1:
        raise DenseRetrieverError("embedding backend must return exactly one query vector")
    vector = vectors[0]
    if not isinstance(vector, list) or not vector:
        raise DenseRetrieverError("query vector must be a non-empty list")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in vector):
        raise DenseRetrieverError("query vector must contain only numbers")
    return [float(value) for value in vector]


def _to_retrieval_result(hit: VectorQueryResult) -> RetrievalResult:
    if not isinstance(hit, VectorQueryResult):
        raise DenseRetrieverError("vector store must return VectorQueryResult objects")
    return RetrievalResult(
        chunk_id=hit.id,
        score=float(hit.score),
        text=hit.text,
        metadata=dict(hit.metadata),
    )


def _record_trace(trace: Any | None, name: str, data: dict[str, Any]) -> None:
    if hasattr(trace, "record_stage"):
        trace.record_stage(name, data)
