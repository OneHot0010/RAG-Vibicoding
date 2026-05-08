"""Hybrid retrieval orchestration."""

from __future__ import annotations

from typing import Any, Mapping

from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.fusion import RRFusion
from core.query_engine.query_processor import QueryProcessor
from core.query_engine.sparse_retriever import SparseRetriever
from core.settings import Settings
from core.types import RetrievalResult


class HybridSearchError(ValueError):
    """Raised when hybrid search input or configuration is invalid."""


class HybridSearch:
    """Run query processing, dense retrieval, sparse retrieval, fusion, and filtering."""

    def __init__(
        self,
        settings: Settings,
        query_processor: QueryProcessor | None = None,
        dense_retriever: Any | None = None,
        sparse_retriever: Any | None = None,
        fusion: RRFusion | None = None,
    ) -> None:
        self.settings = settings
        self.query_processor = query_processor or QueryProcessor()
        self.dense_retriever = dense_retriever or DenseRetriever(settings)
        self.sparse_retriever = sparse_retriever or SparseRetriever(settings)
        self.fusion = fusion or RRFusion()

    def search(
        self,
        query: str,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        """Return fused retrieval results for a query."""
        if top_k <= 0:
            raise HybridSearchError("top_k must be greater than 0")
        processed = self.query_processor.process(query, filters=filters, trace=trace)
        dense_top_k = _configured_top_k(self.settings.retrieval.top_k_dense, top_k)
        sparse_top_k = _configured_top_k(self.settings.retrieval.top_k_sparse, top_k)

        dense_results = self._retrieve_dense(processed.normalized_query or processed.original_query, dense_top_k, processed.filters, trace)
        sparse_results = self._retrieve_sparse(processed.keywords, sparse_top_k, trace)
        if not dense_results and not sparse_results:
            _record_trace(trace, "hybrid_search.completed", {"result_count": 0, "fallback": "none"})
            return []

        fused = self.fusion.fuse(dense_results, sparse_results, top_k=top_k, trace=trace)
        filtered = self._apply_metadata_filters(fused, processed.filters)[:top_k]
        _record_trace(
            trace,
            "hybrid_search.completed",
            {
                "dense_count": len(dense_results),
                "sparse_count": len(sparse_results),
                "result_count": len(filtered),
                "filters": processed.filters,
            },
        )
        return filtered

    def _apply_metadata_filters(
        self,
        candidates: list[RetrievalResult],
        filters: Mapping[str, Any] | None,
    ) -> list[RetrievalResult]:
        """Apply metadata filters after fusion as a defensive fallback."""
        _validate_results(candidates)
        if not filters:
            return candidates
        normalized_filters = dict(filters)
        return [
            candidate
            for candidate in candidates
            if all(candidate.metadata.get(key) == value for key, value in normalized_filters.items())
        ]

    def _retrieve_dense(
        self,
        query: str,
        top_k: int,
        filters: Mapping[str, Any],
        trace: Any | None,
    ) -> list[RetrievalResult]:
        try:
            results = self.dense_retriever.retrieve(query, top_k=top_k, filters=filters, trace=trace)
            _validate_results(results)
            return results
        except Exception as exc:
            _record_trace(trace, "hybrid_search.route_failed", {"route": "dense", "error": str(exc)})
            return []

    def _retrieve_sparse(self, keywords: list[str], top_k: int, trace: Any | None) -> list[RetrievalResult]:
        try:
            results = self.sparse_retriever.retrieve(keywords, top_k=top_k, trace=trace)
            _validate_results(results)
            return results
        except Exception as exc:
            _record_trace(trace, "hybrid_search.route_failed", {"route": "sparse", "error": str(exc)})
            return []


def _configured_top_k(configured: int, fallback: int) -> int:
    try:
        value = int(configured)
    except (TypeError, ValueError):
        return fallback
    return max(value, fallback)


def _validate_results(results: list[RetrievalResult]) -> None:
    if not isinstance(results, list):
        raise HybridSearchError("retriever results must be a list")
    for result in results:
        if not isinstance(result, RetrievalResult):
            raise HybridSearchError("retriever results must contain RetrievalResult objects")


def _record_trace(trace: Any | None, name: str, data: dict[str, Any]) -> None:
    if hasattr(trace, "record_stage"):
        trace.record_stage(name, data)
