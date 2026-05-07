"""Contract tests for vector store abstractions and factory routing."""

from __future__ import annotations

from math import sqrt
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
from libs.vector_store import (
    BaseVectorStore,
    VectorQueryResult,
    VectorRecord,
    VectorStoreFactory,
    VectorStoreFactoryError,
)


class FakeVectorStore(BaseVectorStore):
    def __init__(self, settings: VectorStoreSettings) -> None:
        self.settings = settings
        self.records: dict[str, VectorRecord] = {}

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
        filters = filters or {}
        scored: list[VectorQueryResult] = []
        for record in self.records.values():
            if any(record.metadata.get(key) != value for key, value in filters.items()):
                continue
            scored.append(
                VectorQueryResult(
                    id=record.id,
                    score=_cosine_similarity(vector, record.vector),
                    text=record.text,
                    metadata=record.metadata,
                )
            )
        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]


class NotAVectorStore:
    pass


@pytest.fixture(autouse=True)
def clear_factory_registry() -> None:
    VectorStoreFactory.clear()


def make_settings(backend: str = "fake") -> Settings:
    return Settings(
        llm=LLMSettings(provider="fake", model="fake-chat"),
        embedding=EmbeddingSettings(provider="fake", model="fake-embedding"),
        vision_llm=VisionLLMSettings(provider="fake", model="fake-vision"),
        splitter=SplitterSettings(strategy="fake", chunk_size=100, chunk_overlap=10),
        vector_store=VectorStoreSettings(backend=backend, persist_path="./tmp/vector"),
        retrieval=RetrievalSettings(
            sparse_backend="bm25",
            fusion_algorithm="rrf",
            top_k_dense=20,
            top_k_sparse=20,
            top_k_final=10,
        ),
        rerank=RerankSettings(backend="none"),
        evaluation=EvaluationSettings(backends=["custom"], golden_test_set="./golden.json"),
        observability=ObservabilitySettings(enabled=True, log_file="./logs/traces.jsonl"),
    )


def test_vector_record_contract_shape() -> None:
    record = VectorRecord(
        id="chunk-1",
        vector=[1.0, 0.0],
        text="hello",
        metadata={"source": "fixture.md"},
    )

    assert record.id == "chunk-1"
    assert record.vector == [1.0, 0.0]
    assert record.text == "hello"
    assert record.metadata["source"] == "fixture.md"


def test_vector_store_upsert_and_query_contract() -> None:
    store = FakeVectorStore(VectorStoreSettings(backend="fake", persist_path="./tmp/vector"))
    store.upsert(
        [
            VectorRecord(id="a", vector=[1.0, 0.0], text="alpha", metadata={"collection": "docs"}),
            VectorRecord(id="b", vector=[0.0, 1.0], text="beta", metadata={"collection": "docs"}),
            VectorRecord(id="c", vector=[1.0, 0.0], text="gamma", metadata={"collection": "other"}),
        ]
    )

    results = store.query([1.0, 0.0], top_k=2, filters={"collection": "docs"})

    assert results == [
        VectorQueryResult(id="a", score=1.0, text="alpha", metadata={"collection": "docs"}),
        VectorQueryResult(id="b", score=0.0, text="beta", metadata={"collection": "docs"}),
    ]


def test_upsert_is_idempotent_by_record_id() -> None:
    store = FakeVectorStore(VectorStoreSettings(backend="fake", persist_path="./tmp/vector"))
    store.upsert([VectorRecord(id="a", vector=[1.0], text="old")])
    store.upsert([VectorRecord(id="a", vector=[2.0], text="new")])

    results = store.query([2.0], top_k=1)

    assert len(store.records) == 1
    assert results[0].text == "new"


def test_factory_routes_using_full_settings() -> None:
    VectorStoreFactory.register("fake", FakeVectorStore)

    store = VectorStoreFactory.create(make_settings(backend="fake"))

    assert isinstance(store, FakeVectorStore)
    assert store.settings.persist_path == "./tmp/vector"


def test_factory_routes_using_vector_settings_directly() -> None:
    VectorStoreFactory.register("fake", FakeVectorStore)

    store = VectorStoreFactory.create(VectorStoreSettings(backend="fake", persist_path="./direct"))

    assert isinstance(store, FakeVectorStore)
    assert store.settings.persist_path == "./direct"


def test_factory_normalizes_backend_names() -> None:
    VectorStoreFactory.register("fake", FakeVectorStore)

    store = VectorStoreFactory.create(VectorStoreSettings(backend=" FAKE ", persist_path="./tmp/vector"))

    assert isinstance(store, FakeVectorStore)


def test_factory_unknown_backend_has_readable_error() -> None:
    VectorStoreFactory.register("fake", FakeVectorStore)

    with pytest.raises(VectorStoreFactoryError, match="Unknown vector store backend: missing"):
        VectorStoreFactory.create(VectorStoreSettings(backend="missing", persist_path="./tmp/vector"))


def test_register_requires_callable_builder() -> None:
    with pytest.raises(TypeError, match="builder must be callable"):
        VectorStoreFactory.register("bad", "not-callable")  # type: ignore[arg-type]


def test_factory_rejects_builder_returning_wrong_type() -> None:
    VectorStoreFactory.register("bad", lambda settings: NotAVectorStore())  # type: ignore[return-value]

    with pytest.raises(VectorStoreFactoryError, match="expected BaseVectorStore"):
        VectorStoreFactory.create(VectorStoreSettings(backend="bad", persist_path="./tmp/vector"))


def test_unregister_removes_backend() -> None:
    VectorStoreFactory.register("fake", FakeVectorStore)
    VectorStoreFactory.unregister("fake")

    with pytest.raises(VectorStoreFactoryError, match="Registered backends: none"):
        VectorStoreFactory.create(VectorStoreSettings(backend="fake", persist_path="./tmp/vector"))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
