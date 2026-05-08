"""Integration tests for hybrid dense/sparse search orchestration."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from core.query_engine import HybridSearch, HybridSearchError, QueryProcessor, RRFusion
from core.settings import (
    EmbeddingSettings,
    EvaluationSettings,
    LLMSettings,
    ObservabilitySettings,
    RerankSettings,
    RetrievalSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
    VisionLLMSettings,
)
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult


class FakeDenseRetriever:
    def __init__(self, results: list[RetrievalResult] | Exception) -> None:
        self.results = results
        self.calls: list[tuple[str, int, Mapping[str, Any], Any | None]] = []

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append((query, top_k, filters or {}, trace))
        if isinstance(self.results, Exception):
            raise self.results
        return self.results[:top_k]


class FakeSparseRetriever:
    def __init__(self, results: list[RetrievalResult] | Exception) -> None:
        self.results = results
        self.calls: list[tuple[list[str], int, Any | None]] = []

    def retrieve(self, keywords: list[str], top_k: int, trace: Any | None = None) -> list[RetrievalResult]:
        self.calls.append((keywords, top_k, trace))
        if isinstance(self.results, Exception):
            raise self.results
        return self.results[:top_k]


def make_settings() -> Settings:
    return Settings(
        llm=LLMSettings(provider="fake", model="fake-chat"),
        embedding=EmbeddingSettings(provider="fake", model="fake-embedding"),
        vision_llm=VisionLLMSettings(provider="fake", model="fake-vision"),
        splitter=SplitterSettings(strategy="fake", chunk_size=100, chunk_overlap=0),
        vector_store=VectorStoreSettings(backend="fake", persist_path="./tmp/vector"),
        retrieval=RetrievalSettings("bm25", "rrf", 4, 5, 3),
        rerank=RerankSettings(backend="none"),
        evaluation=EvaluationSettings(backends=["custom"], golden_test_set="./golden.json"),
        observability=ObservabilitySettings(enabled=True, log_file="./logs/traces.jsonl"),
    )


def hit(chunk_id: str, score: float, collection: str = "docs", text: str | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text or f"{chunk_id} text",
        metadata={"source_path": f"fixtures/{chunk_id}.md", "collection": collection, "doc_type": "pdf"},
    )


def test_hybrid_search_fuses_dense_and_sparse_results() -> None:
    dense = FakeDenseRetriever([hit("a", 0.9), hit("b", 0.8)])
    sparse = FakeSparseRetriever([hit("b", 3.0), hit("c", 2.0)])

    results = HybridSearch(
        make_settings(),
        query_processor=QueryProcessor(),
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion=RRFusion(k=60),
    ).search("Hybrid retrieval systems", top_k=3, filters={"collection": "docs"})

    assert [result.chunk_id for result in results] == ["b", "a", "c"]
    assert all(result.metadata["collection"] == "docs" for result in results)
    assert results[0].metadata["fusion"]["algorithm"] == "rrf"
    assert dense.calls[0][0] == "Hybrid retrieval systems"
    assert dense.calls[0][1] == 4
    assert dense.calls[0][2] == {"collection": "docs"}
    assert sparse.calls[0][0] == ["hybrid", "retrieval", "systems"]
    assert sparse.calls[0][1] == 5


def test_inline_filters_are_applied_after_fusion_as_fallback() -> None:
    dense = FakeDenseRetriever([hit("a", 0.9, collection="notes"), hit("b", 0.8, collection="docs")])
    sparse = FakeSparseRetriever([hit("c", 3.0, collection="notes"), hit("d", 2.0, collection="docs")])

    results = HybridSearch(make_settings(), dense_retriever=dense, sparse_retriever=sparse).search(
        "retrieval collection:docs",
        top_k=4,
    )

    assert [result.chunk_id for result in results] == ["b", "d"]
    assert dense.calls[0][2] == {"collection": "docs"}


def test_dense_failure_degrades_to_sparse_results() -> None:
    trace = TraceContext()
    dense = FakeDenseRetriever(RuntimeError("dense unavailable"))
    sparse = FakeSparseRetriever([hit("sparse-only", 2.0)])

    results = HybridSearch(make_settings(), dense_retriever=dense, sparse_retriever=sparse).search(
        "retrieval",
        top_k=3,
        trace=trace,
    )

    assert [result.chunk_id for result in results] == ["sparse-only"]
    assert {"name": "hybrid_search.route_failed", "data": {"route": "dense", "error": "dense unavailable"}} in trace.stages
    assert trace.stages[-1]["name"] == "hybrid_search.completed"


def test_sparse_failure_degrades_to_dense_results() -> None:
    dense = FakeDenseRetriever([hit("dense-only", 0.9)])
    sparse = FakeSparseRetriever(RuntimeError("bm25 unavailable"))

    results = HybridSearch(make_settings(), dense_retriever=dense, sparse_retriever=sparse).search(
        "retrieval",
        top_k=3,
    )

    assert [result.chunk_id for result in results] == ["dense-only"]


def test_both_routes_failing_returns_empty_results() -> None:
    trace = TraceContext()

    results = HybridSearch(
        make_settings(),
        dense_retriever=FakeDenseRetriever(RuntimeError("dense failed")),
        sparse_retriever=FakeSparseRetriever(RuntimeError("sparse failed")),
    ).search("retrieval", top_k=2, trace=trace)

    assert results == []
    assert trace.stages[-1] == {"name": "hybrid_search.completed", "data": {"result_count": 0, "fallback": "none"}}


def test_invalid_inputs_have_readable_errors() -> None:
    search = HybridSearch(
        make_settings(),
        dense_retriever=FakeDenseRetriever([]),
        sparse_retriever=FakeSparseRetriever([]),
    )

    with pytest.raises(HybridSearchError, match="top_k"):
        search.search("query", top_k=0)
    with pytest.raises(HybridSearchError, match="RetrievalResult"):
        search._apply_metadata_filters(["bad"], {})  # type: ignore[list-item]
