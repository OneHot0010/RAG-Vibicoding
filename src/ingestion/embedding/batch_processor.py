"""Batch orchestration for dense and sparse chunk encoding."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from core.settings import Settings
from core.types import Chunk, ChunkRecord
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder, SparseEncodingResult


class BatchProcessorError(ValueError):
    """Raised when batch processing configuration or input is invalid."""


@dataclass(frozen=True)
class ProcessedBatch:
    """Encoding output and timing for one stable chunk batch."""

    index: int
    chunks: list[Chunk]
    dense_records: list[ChunkRecord]
    sparse_result: SparseEncodingResult
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize the batch result using stable field names."""
        return {
            "index": self.index,
            "chunk_ids": [chunk.id for chunk in self.chunks],
            "dense_records": [record.to_dict() for record in self.dense_records],
            "sparse_result": self.sparse_result.to_dict(),
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class BatchProcessingResult:
    """Full batch processing result for dense and sparse encoding."""

    batches: list[ProcessedBatch]

    @property
    def dense_records(self) -> list[ChunkRecord]:
        """Return dense records flattened in original chunk order."""
        return [record for batch in self.batches for record in batch.dense_records]

    @property
    def sparse_records(self) -> list[ChunkRecord]:
        """Return sparse records flattened in original chunk order."""
        return [record for batch in self.batches for record in batch.sparse_result.records]

    def to_dict(self) -> dict[str, Any]:
        """Serialize all batches for trace/debug output."""
        return {"batches": [batch.to_dict() for batch in self.batches]}


class BatchProcessor:
    """Split chunks into batches and drive dense/sparse encoders."""

    def __init__(
        self,
        settings: Settings,
        batch_size: int | None = None,
        dense_encoder: DenseEncoder | None = None,
        sparse_encoder: SparseEncoder | None = None,
    ) -> None:
        self.settings = settings
        self.batch_size = _batch_size_from_settings(settings) if batch_size is None else batch_size
        if self.batch_size <= 0:
            raise BatchProcessorError("batch_size must be greater than 0")
        self.dense_encoder = dense_encoder or DenseEncoder(settings)
        self.sparse_encoder = sparse_encoder or SparseEncoder()

    def iter_batches(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        """Split chunks into stable sequential batches."""
        _validate_chunks(chunks)
        return [chunks[index : index + self.batch_size] for index in range(0, len(chunks), self.batch_size)]

    def process(self, chunks: list[Chunk], trace: Any | None = None) -> BatchProcessingResult:
        """Encode chunks batch-by-batch with dense and sparse encoders."""
        batches: list[ProcessedBatch] = []
        for index, batch_chunks in enumerate(self.iter_batches(chunks)):
            started = time.perf_counter()
            dense_records = self.dense_encoder.encode(batch_chunks, trace=trace)
            sparse_result = self.sparse_encoder.encode(batch_chunks, trace=trace)
            duration_ms = (time.perf_counter() - started) * 1000
            batch = ProcessedBatch(
                index=index,
                chunks=batch_chunks,
                dense_records=dense_records,
                sparse_result=sparse_result,
                duration_ms=duration_ms,
            )
            batches.append(batch)
            _record_trace(
                trace,
                "batch_processor.batch",
                {
                    "batch_index": index,
                    "chunk_count": len(batch_chunks),
                    "duration_ms": duration_ms,
                },
            )
        return BatchProcessingResult(batches)


def _batch_size_from_settings(settings: Settings) -> int:
    raw = settings.raw or {}
    ingestion = raw.get("ingestion") if isinstance(raw, dict) else None
    if isinstance(ingestion, dict) and ingestion.get("batch_size") is not None:
        return int(ingestion["batch_size"])
    return 16


def _validate_chunks(chunks: list[Chunk]) -> None:
    if not isinstance(chunks, list):
        raise BatchProcessorError("chunks must be a list")
    for chunk in chunks:
        if not isinstance(chunk, Chunk):
            raise BatchProcessorError("BatchProcessor expects Chunk objects")


def _record_trace(trace: Any | None, name: str, data: dict[str, Any]) -> None:
    if hasattr(trace, "record_stage"):
        trace.record_stage(name, data)
