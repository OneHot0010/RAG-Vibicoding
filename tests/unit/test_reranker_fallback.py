"""Tests for core reranker orchestration and fallback behavior."""

from __future__ import annotations

from typing import Any

import pytest

from core.query_engine import CoreReranker, CoreRerankerError
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
from libs.reranker import BaseReranker, RerankCandidate, RerankResult, RerankerFactory


class FakeReranker(BaseReranker):
    def __init__(self, results: list[RerankResult] | Exception) -> None:
        self.results = results
        self.calls: list[tuple[str, list[RerankCandidate], Any | None]] = []

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Any | None = None,
    ) -> list[RerankResult]:
        self.calls.append((query, candidates, trace))
        if isinstance(self.results, Exception):
            raise self.results
        return self.results


class FactoryReranker(FakeReranker):
    def __init__(self, settings: RerankSettings) -> None:
        super().__init__([])
        self.settings = settings


def make_settings(backend: str = "none") -> Settings:
    return Settings(
        llm=LLMSettings(provider="fake", model="fake-chat"),
        embedding=EmbeddingSettings(provider="fake", model="fake-embedding"),
        vision_llm=VisionLLMSettings(provider="fake", model="fake-vision"),
        splitter=SplitterSettings(strategy="fake", chunk_size=100, chunk_overlap=0),
        vector_store=VectorStoreSettings(backend="fake", persist_path="./tmp/vector"),
        retrieval=RetrievalSettings("bm25", "rrf", 20, 20, 10),
        rerank=RerankSettings(backend=backend, model="fake-reranker", top_m=30),
        evaluation=EvaluationSettings(backends=["custom"], golden_test_set="./golden.json"),
        observability=ObservabilitySettings(enabled=True, log_file="./logs/traces.jsonl"),
    )


def result(chunk_id: str, score: float = 0.1) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=f"{chunk_id} text",
        metadata={"source_path": f"docs/{chunk_id}.md", "fusion": {"score": score}},
    )


def test_core_reranker_converts_candidates_and_applies_backend_order() -> None:
    trace = TraceContext()
    backend = FakeReranker(
        [
            RerankResult("b", "b text", 9.0, 1, {"source_path": "docs/b.md"}),
            RerankResult("a", "a text", 4.0, 2, {"source_path": "docs/a.md"}),
        ]
    )

    results = CoreReranker(make_settings(), reranker=backend).rerank(
        "rerank query",
        [result("a", 0.2), result("b", 0.1)],
        trace=trace,
    )

    assert [item.chunk_id for item in results] == ["b", "a"]
    assert [item.score for item in results] == [9.0, 4.0]
    assert results[0].metadata["rerank"] == {"fallback": False, "rank": 1, "score": 9.0}
    assert backend.calls[0][0] == "rerank query"
    assert [candidate.id for candidate in backend.calls[0][1]] == ["a", "b"]
    assert trace.stages == [
        {
            "name": "core_reranker.rerank",
            "data": {"candidate_count": 2, "result_count": 2, "fallback": False},
        }
    ]


def test_top_k_limits_candidates_before_backend_call() -> None:
    backend = FakeReranker([RerankResult("a", "a text", 1.0, 1, {"source_path": "docs/a.md"})])

    results = CoreReranker(make_settings(), reranker=backend).rerank(
        "query",
        [result("a"), result("b")],
        top_k=1,
    )

    assert [candidate.id for candidate in backend.calls[0][1]] == ["a"]
    assert [item.chunk_id for item in results] == ["a"]


def test_backend_exception_falls_back_to_fusion_order_and_marks_metadata() -> None:
    trace = TraceContext()
    candidates = [result("a", 0.3), result("b", 0.2)]

    results = CoreReranker(make_settings(), reranker=FakeReranker(RuntimeError("reranker down"))).rerank(
        "query",
        candidates,
        trace=trace,
    )

    assert [item.chunk_id for item in results] == ["a", "b"]
    assert [item.score for item in results] == [0.3, 0.2]
    assert all(item.metadata["rerank"]["fallback"] is True for item in results)
    assert results[0].metadata["rerank"]["reason"] == "reranker down"
    assert trace.stages == [
        {
            "name": "core_reranker.fallback",
            "data": {"candidate_count": 2, "result_count": 2, "error": "reranker down"},
        }
    ]


def test_invalid_backend_results_fall_back() -> None:
    candidates = [result("a"), result("b")]
    duplicate = FakeReranker(
        [
            RerankResult("a", "a text", 1.0, 1, {"source_path": "docs/a.md"}),
            RerankResult("a", "a text", 0.5, 2, {"source_path": "docs/a.md"}),
        ]
    )

    duplicate_results = CoreReranker(make_settings(), reranker=duplicate).rerank("query", candidates)

    assert [item.chunk_id for item in duplicate_results] == ["a", "b"]
    assert duplicate_results[0].metadata["rerank"]["fallback"] is True

    unknown = FakeReranker([RerankResult("missing", "x", 1.0, 1, {"source_path": "x.md"})])
    unknown_results = CoreReranker(make_settings(), reranker=unknown).rerank("query", candidates)
    assert unknown_results[0].metadata["rerank"]["fallback"] is True


def test_empty_candidates_and_zero_top_k_return_empty() -> None:
    backend = FakeReranker([])
    reranker = CoreReranker(make_settings(), reranker=backend)

    assert reranker.rerank("query", []) == []
    assert reranker.rerank("query", [result("a")], top_k=0) == []
    assert backend.calls == []


def test_reranker_can_be_created_from_factory() -> None:
    RerankerFactory.register("factory-reranker", FactoryReranker)

    reranker = CoreReranker(make_settings(backend="factory-reranker"))

    assert isinstance(reranker.reranker, FactoryReranker)
    assert reranker.reranker.settings.model == "fake-reranker"


def test_invalid_inputs_have_readable_errors() -> None:
    reranker = CoreReranker(make_settings(), reranker=FakeReranker([]))

    with pytest.raises(CoreRerankerError, match="query"):
        reranker.rerank("", [result("a")])
    with pytest.raises(CoreRerankerError, match="candidates"):
        reranker.rerank("query", ["bad"])  # type: ignore[list-item]
    with pytest.raises(CoreRerankerError, match="top_k"):
        reranker.rerank("query", [result("a")], top_k=-1)
