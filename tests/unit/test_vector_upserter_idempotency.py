"""Tests for idempotent vector upsert behavior."""

from __future__ import annotations

from typing import Any, Mapping

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
from core.trace.trace_context import TraceContext
from core.types import ChunkRecord
from ingestion.storage import VectorUpserter, VectorUpserterError
from libs.vector_store import BaseVectorStore, VectorQueryResult, VectorRecord, VectorStoreFactory


class FakeVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self.records: dict[str, VectorRecord] = {}
        self.calls: list[tuple[list[VectorRecord], Any | None]] = []

    def upsert(self, records: list[VectorRecord], trace: Any | None = None) -> None:
        self.calls.append((records, trace))
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
        return [self.records[record_id] for record_id in ids if record_id in self.records]


class FactoryVectorStore(FakeVectorStore):
    def __init__(self, settings: VectorStoreSettings) -> None:
        super().__init__()
        self.settings = settings


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


def record(text: str = "alpha", chunk_index: int = 0, vector: list[float] | None = None) -> ChunkRecord:
    return ChunkRecord(
        id=f"original-{chunk_index}",
        text=text,
        metadata={"source_path": "docs/a.md", "chunk_index": chunk_index, "collection": "docs"},
        dense_vector=vector or [1.0, 0.0],
    )


def test_upsert_writes_vector_records_to_store() -> None:
    store = FakeVectorStore()
    upserter = VectorUpserter(make_settings(), vector_store=store)

    written = upserter.upsert([record("alpha", 0), record("beta", 1)])

    assert [item.text for item in written] == ["alpha", "beta"]
    assert len(store.calls) == 1
    assert list(store.records) == [written[0].id, written[1].id]
    assert written[0].metadata["original_chunk_id"] == "original-0"
    assert written[0].metadata["content_hash"]


def test_same_chunk_upsert_is_idempotent() -> None:
    store = FakeVectorStore()
    upserter = VectorUpserter(make_settings(), vector_store=store)
    first = record("same content", 0)
    second = record("same content", 0)

    first_written = upserter.upsert([first])[0]
    second_written = upserter.upsert([second])[0]

    assert first_written.id == second_written.id
    assert len(store.records) == 1


def test_content_change_changes_vector_id() -> None:
    upserter = VectorUpserter(make_settings(), vector_store=FakeVectorStore())

    old_id = upserter.to_vector_record(record("old content", 0)).id
    new_id = upserter.to_vector_record(record("new content", 0)).id

    assert old_id != new_id


def test_batch_upsert_preserves_order() -> None:
    upserter = VectorUpserter(make_settings(), vector_store=FakeVectorStore())
    records = [record("a", 0), record("b", 1), record("c", 2)]

    written = upserter.upsert(records)

    assert [item.metadata["original_chunk_id"] for item in written] == ["original-0", "original-1", "original-2"]


def test_vector_store_can_be_created_from_factory() -> None:
    VectorStoreFactory.clear()
    VectorStoreFactory.register("factory", FactoryVectorStore)

    upserter = VectorUpserter(make_settings(backend="factory"))

    assert isinstance(upserter.vector_store, FactoryVectorStore)
    assert upserter.vector_store.settings.persist_path == "./tmp/vector"


def test_empty_upsert_does_not_call_store() -> None:
    store = FakeVectorStore()

    assert VectorUpserter(make_settings(), vector_store=store).upsert([]) == []
    assert store.calls == []


def test_trace_is_passed_to_store_and_records_stage() -> None:
    store = FakeVectorStore()
    trace = TraceContext()

    VectorUpserter(make_settings(), vector_store=store).upsert([record()], trace=trace)

    assert store.calls[0][1] is trace
    assert trace.stages == [{"name": "vector_upserter.upsert", "data": {"record_count": 1}}]


def test_missing_dense_vector_is_rejected() -> None:
    bad = ChunkRecord(id="bad", text="alpha", metadata={"source_path": "docs/a.md", "chunk_index": 0})

    with pytest.raises(VectorUpserterError, match="missing dense_vector"):
        VectorUpserter(make_settings(), vector_store=FakeVectorStore()).upsert([bad])


def test_missing_chunk_index_is_rejected() -> None:
    upserter = VectorUpserter(make_settings(), vector_store=FakeVectorStore())

    with pytest.raises(VectorUpserterError, match="chunk_index"):
        upserter.upsert([ChunkRecord(id="bad", text="alpha", metadata={"source_path": "docs/a.md"}, dense_vector=[1.0])])


def test_non_chunk_record_input_is_rejected() -> None:
    with pytest.raises(VectorUpserterError, match="ChunkRecord"):
        VectorUpserter(make_settings(), vector_store=FakeVectorStore()).upsert(["not record"])  # type: ignore[list-item]
