"""Tests for batch encoding orchestration."""

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
from ingestion.embedding import (
    BatchProcessingResult,
    BatchProcessor,
    BatchProcessorError,
    SparseEncodingResult,
)


class FakeDenseEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, chunks: list[Chunk], trace: Any | None = None) -> list[ChunkRecord]:
        self.calls.append([chunk.id for chunk in chunks])
        return [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata,
                dense_vector=[float(index), float(len(chunk.text))],
            )
            for index, chunk in enumerate(chunks)
        ]


class FakeSparseEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, chunks: list[Chunk], trace: Any | None = None) -> SparseEncodingResult:
        self.calls.append([chunk.id for chunk in chunks])
        records = [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata={**chunk.metadata, "doc_length": 1},
                sparse_vector={chunk.id: 1.0},
            )
            for chunk in chunks
        ]
        return SparseEncodingResult(records=records, document_frequency={chunk.id: 1 for chunk in chunks}, total_chunks=len(chunks), average_doc_length=1.0 if chunks else 0.0)


def make_settings(batch_size: int | None = None) -> Settings:
    raw = {"ingestion": {}} if batch_size is None else {"ingestion": {"batch_size": batch_size}}
    return Settings(
        llm=LLMSettings(provider="fake", model="fake-chat"),
        embedding=EmbeddingSettings(provider="fake", model="fake-embedding"),
        vision_llm=VisionLLMSettings(provider="fake", model="fake-vision"),
        splitter=SplitterSettings(strategy="fake", chunk_size=100, chunk_overlap=0),
        vector_store=VectorStoreSettings(backend="fake", persist_path="./tmp/vector"),
        retrieval=RetrievalSettings("bm25", "rrf", 20, 20, 10),
        rerank=RerankSettings(backend="none"),
        evaluation=EvaluationSettings(backends=["custom"], golden_test_set="./golden.json"),
        observability=ObservabilitySettings(enabled=True, log_file="./logs/traces.jsonl"),
        raw=raw,
    )


def chunks(count: int = 5) -> list[Chunk]:
    return [
        Chunk(
            id=f"chunk-{index}",
            text=f"text {index}",
            metadata={"source_path": "docs/a.md", "chunk_index": index},
            start_offset=index,
            end_offset=index + 1,
            source_ref="doc-1",
        )
        for index in range(count)
    ]


def processor(batch_size: int = 2) -> tuple[BatchProcessor, FakeDenseEncoder, FakeSparseEncoder]:
    dense = FakeDenseEncoder()
    sparse = FakeSparseEncoder()
    return (
        BatchProcessor(make_settings(), batch_size=batch_size, dense_encoder=dense, sparse_encoder=sparse),  # type: ignore[arg-type]
        dense,
        sparse,
    )


def test_iter_batches_splits_five_chunks_into_three_stable_batches() -> None:
    batch_processor, _, _ = processor(batch_size=2)

    batches = batch_processor.iter_batches(chunks(5))

    assert [[chunk.id for chunk in batch] for batch in batches] == [
        ["chunk-0", "chunk-1"],
        ["chunk-2", "chunk-3"],
        ["chunk-4"],
    ]


def test_process_drives_dense_and_sparse_per_batch() -> None:
    batch_processor, dense, sparse = processor(batch_size=2)

    result = batch_processor.process(chunks(5))

    assert isinstance(result, BatchProcessingResult)
    assert [batch.index for batch in result.batches] == [0, 1, 2]
    assert dense.calls == [["chunk-0", "chunk-1"], ["chunk-2", "chunk-3"], ["chunk-4"]]
    assert sparse.calls == [["chunk-0", "chunk-1"], ["chunk-2", "chunk-3"], ["chunk-4"]]


def test_flattened_records_preserve_original_order() -> None:
    batch_processor, _, _ = processor(batch_size=2)

    result = batch_processor.process(chunks(5))

    assert [record.id for record in result.dense_records] == [f"chunk-{index}" for index in range(5)]
    assert [record.id for record in result.sparse_records] == [f"chunk-{index}" for index in range(5)]


def test_batch_size_can_come_from_settings_raw() -> None:
    dense = FakeDenseEncoder()
    sparse = FakeSparseEncoder()
    batch_processor = BatchProcessor(
        make_settings(batch_size=3),
        dense_encoder=dense,  # type: ignore[arg-type]
        sparse_encoder=sparse,  # type: ignore[arg-type]
    )

    assert [[chunk.id for chunk in batch] for batch in batch_processor.iter_batches(chunks(5))] == [
        ["chunk-0", "chunk-1", "chunk-2"],
        ["chunk-3", "chunk-4"],
    ]


def test_empty_chunks_return_empty_result() -> None:
    batch_processor, dense, sparse = processor(batch_size=2)

    result = batch_processor.process([])

    assert result == BatchProcessingResult([])
    assert dense.calls == []
    assert sparse.calls == []


def test_trace_records_each_batch() -> None:
    batch_processor, _, _ = processor(batch_size=2)
    trace = TraceContext()

    batch_processor.process(chunks(3), trace=trace)

    batch_stages = [stage for stage in trace.stages if stage["name"] == "batch_processor.batch"]
    assert [stage["data"]["batch_index"] for stage in batch_stages] == [0, 1]
    assert [stage["data"]["chunk_count"] for stage in batch_stages] == [2, 1]
    assert all(stage["data"]["duration_ms"] >= 0 for stage in batch_stages)


def test_result_serializes_stable_shape() -> None:
    batch_processor, _, _ = processor(batch_size=2)

    serialized = batch_processor.process(chunks(1)).to_dict()

    assert serialized["batches"][0]["index"] == 0
    assert serialized["batches"][0]["chunk_ids"] == ["chunk-0"]
    assert serialized["batches"][0]["dense_records"][0]["id"] == "chunk-0"
    assert serialized["batches"][0]["sparse_result"]["total_chunks"] == 1


def test_invalid_batch_size_is_rejected() -> None:
    with pytest.raises(BatchProcessorError, match="batch_size"):
        BatchProcessor(make_settings(), batch_size=0, dense_encoder=FakeDenseEncoder(), sparse_encoder=FakeSparseEncoder())  # type: ignore[arg-type]


def test_non_chunk_input_is_rejected() -> None:
    batch_processor, _, _ = processor(batch_size=2)

    with pytest.raises(BatchProcessorError, match="Chunk objects"):
        batch_processor.process(["not chunk"])  # type: ignore[list-item]
