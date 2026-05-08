"""Tests for dense semantic retriever orchestration."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from core.query_engine import DenseRetriever, DenseRetrieverError
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
from libs.embedding import BaseEmbedding, EmbeddingFactory
from libs.vector_store import BaseVectorStore, VectorQueryResult, VectorRecord, VectorStoreFactory


class FakeEmbedding(BaseEmbedding):
    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.vectors = vectors or [[1.0, 0.0]]
        self.calls: list[tuple[list[str], Any | None]] = []

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        self.calls.append((texts, trace))
        return self.vectors


class FactoryEmbedding(FakeEmbedding):
    def __init__(self, settings: EmbeddingSettings) -> None:
        super().__init__()
        self.settings = settings


class FakeVectorStore(BaseVectorStore):
    def __init__(self, hits: list[VectorQueryResult] | None = None) -> None:
        self.hits = hits or [
            VectorQueryResult(
                id="chunk-a",
                score=0.9,
                text="alpha text",
                metadata={"source_path": "docs/a.md", "collection": "docs"},
            )
        ]
        self.calls: list[tuple[list[float], int, Mapping[str, Any], Any | None]] = []

    def upsert(self, records: list[VectorRecord], trace: Any | None = None) -> None:
        pass

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[VectorQueryResult]:
        self.calls.append((vector, top_k, filters or {}, trace))
        return self.hits[:top_k]

    def get_by_ids(self, ids: list[str], trace: Any | None = None) -> list[VectorRecord]:
        return []


class FactoryVectorStore(FakeVectorStore):
    def __init__(self, settings: VectorStoreSettings) -> None:
        super().__init__()
        self.settings = settings


def make_settings(embedding_provider: str = "fake", vector_backend: str = "fake") -> Settings:
    return Settings(
        llm=LLMSettings(provider="fake", model="fake-chat"),
        embedding=EmbeddingSettings(provider=embedding_provider, model="fake-embedding"),
        vision_llm=VisionLLMSettings(provider="fake", model="fake-vision"),
        splitter=SplitterSettings(strategy="fake", chunk_size=100, chunk_overlap=0),
        vector_store=VectorStoreSettings(backend=vector_backend, persist_path="./tmp/vector"),
        retrieval=RetrievalSettings("bm25", "rrf", 20, 20, 10),
        rerank=RerankSettings(backend="none"),
        evaluation=EvaluationSettings(backends=["custom"], golden_test_set="./golden.json"),
        observability=ObservabilitySettings(enabled=True, log_file="./logs/traces.jsonl"),
    )


def test_retrieve_embeds_query_and_calls_vector_store_with_filters() -> None:
    embedding = FakeEmbedding([[0.25, 0.75]])
    store = FakeVectorStore()
    trace = TraceContext()

    results = DenseRetriever(make_settings(), embedding_client=embedding, vector_store=store).retrieve(
        "  hybrid retrieval  ",
        top_k=5,
        filters={"collection": "docs"},
        trace=trace,
    )

    assert embedding.calls == [(["hybrid retrieval"], trace)]
    assert store.calls == [([0.25, 0.75], 5, {"collection": "docs"}, trace)]
    assert results == [
        RetrievalResult(
            chunk_id="chunk-a",
            score=0.9,
            text="alpha text",
            metadata={"source_path": "docs/a.md", "collection": "docs"},
        )
    ]
    assert trace.stages[-1] == {
        "name": "dense_retriever.retrieve",
        "data": {"top_k": 5, "result_count": 1, "filters": {"collection": "docs"}},
    }


def test_retrieve_respects_top_k_and_preserves_hit_order() -> None:
    store = FakeVectorStore(
        [
            VectorQueryResult("a", 0.8, "a text", {"source_path": "a.md"}),
            VectorQueryResult("b", 0.7, "b text", {"source_path": "b.md"}),
        ]
    )

    results = DenseRetriever(make_settings(), embedding_client=FakeEmbedding(), vector_store=store).retrieve("query", top_k=1)

    assert [result.chunk_id for result in results] == ["a"]


def test_factories_can_create_dependencies() -> None:
    EmbeddingFactory.register("factory-embedding", FactoryEmbedding)
    VectorStoreFactory.register("factory-store", FactoryVectorStore)

    retriever = DenseRetriever(make_settings("factory-embedding", "factory-store"))

    assert isinstance(retriever.embedding_client, FactoryEmbedding)
    assert isinstance(retriever.vector_store, FactoryVectorStore)
    assert retriever.embedding_client.settings.model == "fake-embedding"
    assert retriever.vector_store.settings.persist_path == "./tmp/vector"


def test_invalid_inputs_have_readable_errors() -> None:
    retriever = DenseRetriever(make_settings(), embedding_client=FakeEmbedding(), vector_store=FakeVectorStore())

    with pytest.raises(DenseRetrieverError, match="query"):
        retriever.retrieve("", top_k=1)
    with pytest.raises(DenseRetrieverError, match="top_k"):
        retriever.retrieve("query", top_k=0)
    with pytest.raises(DenseRetrieverError, match="filters"):
        retriever.retrieve("query", top_k=1, filters=["bad"])  # type: ignore[arg-type]


def test_embedding_vector_shape_is_validated() -> None:
    retriever = DenseRetriever(make_settings(), embedding_client=FakeEmbedding([]), vector_store=FakeVectorStore())
    retriever.embedding_client.vectors = []  # type: ignore[attr-defined]

    with pytest.raises(DenseRetrieverError, match="exactly one"):
        retriever.retrieve("query", top_k=1)

    retriever.embedding_client.vectors = [["bad"]]  # type: ignore[attr-defined]
    with pytest.raises(DenseRetrieverError, match="numbers"):
        retriever.retrieve("query", top_k=1)


def test_vector_store_result_shape_is_validated() -> None:
    store = FakeVectorStore()
    store.hits = ["not a vector result"]  # type: ignore[list-item]

    with pytest.raises(DenseRetrieverError, match="VectorQueryResult"):
        DenseRetriever(make_settings(), embedding_client=FakeEmbedding(), vector_store=store).retrieve("query", top_k=1)
