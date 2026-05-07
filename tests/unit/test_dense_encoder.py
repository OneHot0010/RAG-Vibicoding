"""Tests for dense chunk encoding."""

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
from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from ingestion.embedding import DenseEncoder, DenseEncoderError
from libs.embedding import BaseEmbedding, EmbeddingFactory


class FakeEmbedding(BaseEmbedding):
    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.vectors = vectors
        self.calls: list[tuple[list[str], Any | None]] = []

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        self.calls.append((texts, trace))
        if self.vectors is not None:
            return self.vectors
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]


class FactoryEmbedding(BaseEmbedding):
    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        return [[1.0, 2.0] for _ in texts]


def make_settings(provider: str = "fake") -> Settings:
    return Settings(
        llm=LLMSettings(provider="fake", model="fake-chat"),
        embedding=EmbeddingSettings(provider=provider, model="fake-embedding"),
        vision_llm=VisionLLMSettings(provider="fake", model="fake-vision"),
        splitter=SplitterSettings(strategy="fake", chunk_size=100, chunk_overlap=0),
        vector_store=VectorStoreSettings(backend="fake", persist_path="./tmp/vector"),
        retrieval=RetrievalSettings("bm25", "rrf", 20, 20, 10),
        rerank=RerankSettings(backend="none"),
        evaluation=EvaluationSettings(backends=["custom"], golden_test_set="./golden.json"),
        observability=ObservabilitySettings(enabled=True, log_file="./logs/traces.jsonl"),
    )


def chunks() -> list[Chunk]:
    return [
        Chunk("chunk-1", "alpha", {"source_path": "docs/a.md", "chunk_index": 0}, 0, 5, "doc-1"),
        Chunk("chunk-2", "beta", {"source_path": "docs/a.md", "chunk_index": 1}, 6, 10, "doc-1"),
    ]


def test_dense_encoder_returns_one_record_per_chunk() -> None:
    embedding = FakeEmbedding()

    records = DenseEncoder(make_settings(), embedding=embedding).encode(chunks())

    assert records == [
        ChunkRecord(
            id="chunk-1",
            text="alpha",
            metadata={"source_path": "docs/a.md", "chunk_index": 0},
            dense_vector=[5.0, 0.0],
        ),
        ChunkRecord(
            id="chunk-2",
            text="beta",
            metadata={"source_path": "docs/a.md", "chunk_index": 1},
            dense_vector=[4.0, 1.0],
        ),
    ]
    assert embedding.calls[0][0] == ["alpha", "beta"]


def test_dense_encoder_uses_embedding_factory_when_not_injected() -> None:
    EmbeddingFactory.clear()
    EmbeddingFactory.register("factory", FactoryEmbedding)

    records = DenseEncoder(make_settings(provider="factory")).encode(chunks())

    assert [record.dense_vector for record in records] == [[1.0, 2.0], [1.0, 2.0]]


def test_dense_encoder_returns_empty_for_empty_chunks() -> None:
    embedding = FakeEmbedding()

    assert DenseEncoder(make_settings(), embedding=embedding).encode([]) == []
    assert embedding.calls == []


def test_dense_encoder_passes_trace_to_embedding_and_records_stage() -> None:
    trace = TraceContext()
    embedding = FakeEmbedding()

    DenseEncoder(make_settings(), embedding=embedding).encode(chunks(), trace=trace)

    assert embedding.calls[0][1] is trace
    assert trace.stages == [
        {"name": "dense_encoder.encoded", "data": {"chunk_count": 2, "dimension": 2}}
    ]


def test_vector_count_mismatch_has_readable_error() -> None:
    with pytest.raises(DenseEncoderError, match="vector count mismatch"):
        DenseEncoder(make_settings(), embedding=FakeEmbedding([[1.0, 2.0]])).encode(chunks())


def test_vector_dimension_mismatch_has_readable_error() -> None:
    with pytest.raises(DenseEncoderError, match="dimension mismatch"):
        DenseEncoder(make_settings(), embedding=FakeEmbedding([[1.0], [1.0, 2.0]])).encode(chunks())


def test_non_numeric_vector_has_readable_error() -> None:
    with pytest.raises(DenseEncoderError, match="list of numbers"):
        DenseEncoder(make_settings(), embedding=FakeEmbedding([[1.0, 2.0], [1.0, "bad"]])).encode(  # type: ignore[list-item]
            chunks()
        )


def test_empty_chunk_text_is_rejected() -> None:
    bad_chunk = Chunk("chunk-empty", " ", {"source_path": "docs/a.md"}, 0, 1)

    with pytest.raises(DenseEncoderError, match="text must be non-empty"):
        DenseEncoder(make_settings(), embedding=FakeEmbedding()).encode([bad_chunk])


def test_non_chunk_input_is_rejected() -> None:
    with pytest.raises(DenseEncoderError, match="Chunk objects"):
        DenseEncoder(make_settings(), embedding=FakeEmbedding()).encode(["not chunk"])  # type: ignore[list-item]
