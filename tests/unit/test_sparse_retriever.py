"""Tests for sparse BM25 retriever hydration."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from core.query_engine import SparseRetriever, SparseRetrieverError
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
from core.types import Chunk, RetrievalResult
from ingestion.embedding import SparseEncoder
from ingestion.storage import BM25Hit, BM25Indexer
from libs.vector_store import BaseVectorStore, VectorQueryResult, VectorRecord, VectorStoreFactory


class FakeVectorStore(BaseVectorStore):
    def __init__(self, records: list[VectorRecord] | None = None) -> None:
        self.records = {record.id: record for record in records or []}
        self.calls: list[tuple[list[str], Any | None]] = []

    def upsert(self, records: list[VectorRecord], trace: Any | None = None) -> None:
        for record in records:
            self.records[record.id] = record

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[VectorQueryResult]:
        return []

    def get_by_ids(self, ids: list[str], trace: Any | None = None) -> list[VectorRecord]:
        self.calls.append((ids, trace))
        lookup = dict(self.records)
        for record in self.records.values():
            original_chunk_id = record.metadata.get("original_chunk_id")
            if isinstance(original_chunk_id, str):
                lookup[original_chunk_id] = record
        return [lookup[record_id] for record_id in ids if record_id in lookup]


class FactoryVectorStore(FakeVectorStore):
    def __init__(self, settings: VectorStoreSettings) -> None:
        super().__init__()
        self.settings = settings


class FakeBM25:
    def __init__(self, hits: list[BM25Hit]) -> None:
        self.hits = hits
        self.calls: list[tuple[list[str], int]] = []

    def query(self, keywords: list[str], top_k: int = 10) -> list[BM25Hit]:
        self.calls.append((keywords, top_k))
        return self.hits[:top_k]


def make_settings(backend: str = "fake") -> Settings:
    return Settings(
        llm=LLMSettings(provider="fake", model="fake-chat"),
        embedding=EmbeddingSettings(provider="fake", model="fake-embedding"),
        vision_llm=VisionLLMSettings(provider="fake", model="fake-vision"),
        splitter=SplitterSettings(strategy="fake", chunk_size=100, chunk_overlap=0),
        vector_store=VectorStoreSettings(backend=backend, persist_path="./tmp/vector"),
        retrieval=RetrievalSettings("bm25", "rrf", 20, 20, 10),
        rerank=RerankSettings(backend="none"),
        evaluation=EvaluationSettings(backends=["custom"], golden_test_set="./golden.json"),
        observability=ObservabilitySettings(enabled=True, log_file="./logs/traces.jsonl"),
    )


def chunks() -> list[Chunk]:
    return [
        Chunk("chunk-a", "alpha beta", {"source_path": "docs/a.md"}, 0, 10, "doc"),
        Chunk("chunk-b", "gamma gamma beta", {"source_path": "docs/a.md"}, 11, 27, "doc"),
        Chunk("chunk-c", "delta alpha", {"source_path": "docs/b.md"}, 0, 11, "doc2"),
    ]


def build_bm25_index() -> BM25Indexer:
    indexer = BM25Indexer(index_dir="./tmp/bm25-test")
    indexer.build(SparseEncoder().encode(chunks()))
    return indexer


def vector_store_for_chunks() -> FakeVectorStore:
    return FakeVectorStore(
        [
            VectorRecord("chunk-a", [1.0], "alpha beta", {"source_path": "docs/a.md", "collection": "docs"}),
            VectorRecord("chunk-b", [1.0], "gamma gamma beta", {"source_path": "docs/a.md", "collection": "docs"}),
            VectorRecord("chunk-c", [1.0], "delta alpha", {"source_path": "docs/b.md", "collection": "docs"}),
        ]
    )


def test_sparse_retriever_queries_bm25_and_hydrates_vector_records() -> None:
    store = vector_store_for_chunks()
    retriever = SparseRetriever(make_settings(), bm25_indexer=build_bm25_index(), vector_store=store)

    results = retriever.retrieve(["gamma"], top_k=2)

    assert [result.chunk_id for result in results] == ["chunk-b"]
    assert results[0].text == "gamma gamma beta"
    assert results[0].metadata["source_path"] == "docs/a.md"
    assert isinstance(results[0], RetrievalResult)
    assert store.calls == [(["chunk-b"], None)]


def test_sparse_retriever_preserves_bm25_order_and_scores() -> None:
    hits = [
        BM25Hit("chunk-b", 2.5, "ignored", {"source_path": "ignored.md"}),
        BM25Hit("chunk-a", 1.5, "ignored", {"source_path": "ignored.md"}),
    ]
    store = vector_store_for_chunks()

    results = SparseRetriever(make_settings(), bm25_indexer=FakeBM25(hits), vector_store=store).retrieve(
        ["beta"], top_k=5
    )

    assert [result.to_dict() for result in results] == [
        {
            "chunk_id": "chunk-b",
            "score": 2.5,
            "text": "gamma gamma beta",
            "metadata": {"source_path": "docs/a.md", "collection": "docs"},
        },
        {
            "chunk_id": "chunk-a",
            "score": 1.5,
            "text": "alpha beta",
            "metadata": {"source_path": "docs/a.md", "collection": "docs"},
        },
    ]


def test_sparse_retriever_can_hydrate_by_original_chunk_id_metadata() -> None:
    hits = [BM25Hit("chunk-a", 1.0, "ignored", {"source_path": "ignored.md"})]
    store = FakeVectorStore(
        [
            VectorRecord(
                "stable-vector-id",
                [1.0],
                "alpha beta",
                {"source_path": "docs/a.md", "original_chunk_id": "chunk-a"},
            )
        ]
    )

    results = SparseRetriever(make_settings(), bm25_indexer=FakeBM25(hits), vector_store=store).retrieve(
        ["alpha"], top_k=1
    )

    assert results[0].chunk_id == "chunk-a"
    assert results[0].text == "alpha beta"


def test_missing_vector_records_are_omitted() -> None:
    hits = [BM25Hit("missing", 1.0, "fallback", {"source_path": "docs/a.md"})]

    assert SparseRetriever(make_settings(), bm25_indexer=FakeBM25(hits), vector_store=FakeVectorStore()).retrieve(
        ["missing"], top_k=1
    ) == []


def test_empty_bm25_hits_do_not_call_vector_store() -> None:
    store = vector_store_for_chunks()
    trace = TraceContext()

    results = SparseRetriever(make_settings(), bm25_indexer=FakeBM25([]), vector_store=store).retrieve(
        ["missing"], top_k=3, trace=trace
    )

    assert results == []
    assert store.calls == []
    assert trace.stages == [{"name": "sparse_retriever.retrieve", "data": {"top_k": 3, "result_count": 0}}]


def test_trace_records_sparse_retrieval_summary() -> None:
    trace = TraceContext()

    SparseRetriever(make_settings(), bm25_indexer=build_bm25_index(), vector_store=vector_store_for_chunks()).retrieve(
        ["gamma"], top_k=2, trace=trace
    )

    assert trace.stages == [
        {
            "name": "sparse_retriever.retrieve",
            "data": {"top_k": 2, "result_count": 1, "keyword_count": 1},
        }
    ]


def test_vector_store_can_be_created_from_factory() -> None:
    VectorStoreFactory.register("factory-store", FactoryVectorStore)

    retriever = SparseRetriever(make_settings(backend="factory-store"), bm25_indexer=FakeBM25([]))

    assert isinstance(retriever.vector_store, FactoryVectorStore)
    assert retriever.vector_store.settings.persist_path == "./tmp/vector"


def test_invalid_inputs_have_readable_errors() -> None:
    retriever = SparseRetriever(make_settings(), bm25_indexer=FakeBM25([]), vector_store=FakeVectorStore())

    with pytest.raises(SparseRetrieverError, match="keywords"):
        retriever.retrieve("alpha", top_k=1)  # type: ignore[arg-type]
    with pytest.raises(SparseRetrieverError, match="keywords"):
        retriever.retrieve([""], top_k=1)
    with pytest.raises(SparseRetrieverError, match="top_k"):
        retriever.retrieve(["alpha"], top_k=0)


def test_backend_shapes_are_validated() -> None:
    with pytest.raises(SparseRetrieverError, match="BM25Hit"):
        SparseRetriever(make_settings(), bm25_indexer=FakeBM25(["bad"]), vector_store=FakeVectorStore()).retrieve(  # type: ignore[list-item]
            ["alpha"], top_k=1
        )

    class BadVectorStore(FakeVectorStore):
        def get_by_ids(self, ids: list[str], trace: Any | None = None) -> list[VectorRecord]:
            return ["bad"]  # type: ignore[list-item,return-value]

    with pytest.raises(SparseRetrieverError, match="VectorRecord"):
        SparseRetriever(
            make_settings(),
            bm25_indexer=FakeBM25([BM25Hit("chunk-a", 1.0, "x", {"source_path": "x.md"})]),
            vector_store=BadVectorStore(),
        ).retrieve(["alpha"], top_k=1)
