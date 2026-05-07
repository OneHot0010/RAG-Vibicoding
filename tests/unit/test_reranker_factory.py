"""Tests for reranker abstraction, none fallback, and factory routing."""

from __future__ import annotations

from typing import Any

import pytest

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
from libs.reranker import (
    BaseReranker,
    NoneReranker,
    RerankCandidate,
    RerankResult,
    RerankerFactory,
    RerankerFactoryError,
)


class FakeReranker(BaseReranker):
    def __init__(self, settings: RerankSettings) -> None:
        self.settings = settings

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Any | None = None,
    ) -> list[RerankResult]:
        ranked = sorted(candidates, key=lambda candidate: candidate.text.count(query), reverse=True)
        return [
            RerankResult(
                id=candidate.id,
                text=candidate.text,
                score=float(candidate.text.count(query)),
                rank=index + 1,
                metadata=candidate.metadata,
            )
            for index, candidate in enumerate(ranked)
        ]


class NotAReranker:
    pass


@pytest.fixture(autouse=True)
def reset_factory_registry() -> None:
    RerankerFactory.reset_defaults()


def make_settings(backend: str = "none") -> Settings:
    return Settings(
        llm=LLMSettings(provider="fake", model="fake-chat"),
        embedding=EmbeddingSettings(provider="fake", model="fake-embedding"),
        vision_llm=VisionLLMSettings(provider="fake", model="fake-vision"),
        splitter=SplitterSettings(strategy="fake", chunk_size=100, chunk_overlap=10),
        vector_store=VectorStoreSettings(backend="fake", persist_path="./tmp/vector"),
        retrieval=RetrievalSettings(
            sparse_backend="bm25",
            fusion_algorithm="rrf",
            top_k_dense=20,
            top_k_sparse=20,
            top_k_final=10,
        ),
        rerank=RerankSettings(backend=backend, model="fake-reranker", top_m=30),
        evaluation=EvaluationSettings(backends=["custom"], golden_test_set="./golden.json"),
        observability=ObservabilitySettings(enabled=True, log_file="./logs/traces.jsonl"),
    )


def test_none_reranker_preserves_order_and_scores() -> None:
    candidates = [
        RerankCandidate(id="a", text="alpha", score=0.2, metadata={"source": "a.md"}),
        RerankCandidate(id="b", text="beta", score=0.9, metadata={"source": "b.md"}),
    ]

    results = NoneReranker().rerank("query", candidates)

    assert results == [
        RerankResult(id="a", text="alpha", score=0.2, rank=1, metadata={"source": "a.md"}),
        RerankResult(id="b", text="beta", score=0.9, rank=2, metadata={"source": "b.md"}),
    ]


def test_factory_returns_none_reranker_by_default_backend() -> None:
    reranker = RerankerFactory.create(RerankSettings(backend="none"))

    assert isinstance(reranker, NoneReranker)


def test_factory_routes_using_full_settings() -> None:
    RerankerFactory.register("fake", FakeReranker)

    reranker = RerankerFactory.create(make_settings(backend="fake"))

    assert isinstance(reranker, FakeReranker)
    assert reranker.settings.model == "fake-reranker"


def test_factory_routes_using_rerank_settings_directly() -> None:
    RerankerFactory.register("fake", FakeReranker)

    reranker = RerankerFactory.create(RerankSettings(backend="fake", model="direct-reranker"))

    assert isinstance(reranker, FakeReranker)
    assert reranker.settings.model == "direct-reranker"


def test_factory_normalizes_backend_names() -> None:
    RerankerFactory.register("fake", FakeReranker)

    reranker = RerankerFactory.create(RerankSettings(backend=" FAKE "))

    assert isinstance(reranker, FakeReranker)


def test_fake_reranker_can_change_order() -> None:
    reranker = FakeReranker(RerankSettings(backend="fake"))
    candidates = [
        RerankCandidate(id="a", text="needle once"),
        RerankCandidate(id="b", text="needle needle twice"),
    ]

    results = reranker.rerank("needle", candidates)

    assert [result.id for result in results] == ["b", "a"]
    assert [result.rank for result in results] == [1, 2]


def test_factory_unknown_backend_has_readable_error() -> None:
    with pytest.raises(RerankerFactoryError, match="Unknown reranker backend: missing"):
        RerankerFactory.create(RerankSettings(backend="missing"))


def test_register_requires_callable_builder() -> None:
    with pytest.raises(TypeError, match="builder must be callable"):
        RerankerFactory.register("bad", "not-callable")  # type: ignore[arg-type]


def test_factory_rejects_builder_returning_wrong_type() -> None:
    RerankerFactory.register("bad", lambda settings: NotAReranker())  # type: ignore[return-value]

    with pytest.raises(RerankerFactoryError, match="expected BaseReranker"):
        RerankerFactory.create(RerankSettings(backend="bad"))


def test_unregister_restores_none_backend_default() -> None:
    RerankerFactory.register("none", FakeReranker)
    RerankerFactory.unregister("none")

    reranker = RerankerFactory.create(RerankSettings(backend="none"))

    assert isinstance(reranker, NoneReranker)
