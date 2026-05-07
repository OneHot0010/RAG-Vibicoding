"""Idempotent vector upsert adapter for dense chunk records."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from core.settings import Settings
from core.types import ChunkRecord
from libs.vector_store import BaseVectorStore, VectorRecord, VectorStoreFactory


class VectorUpserterError(ValueError):
    """Raised when dense records cannot be converted for vector storage."""


class VectorUpserter:
    """Convert dense ChunkRecords to stable VectorRecords and upsert them."""

    def __init__(self, settings: Settings, vector_store: BaseVectorStore | None = None) -> None:
        self.settings = settings
        self.vector_store = vector_store or VectorStoreFactory.create(settings)

    def upsert(self, records: list[ChunkRecord], trace: Any | None = None) -> list[VectorRecord]:
        """Upsert dense chunk records and return the VectorRecords that were written."""
        vector_records = [self.to_vector_record(record) for record in records]
        if vector_records:
            self.vector_store.upsert(vector_records, trace=trace)
        _record_trace(trace, "vector_upserter.upsert", {"record_count": len(vector_records)})
        return vector_records

    def to_vector_record(self, record: ChunkRecord) -> VectorRecord:
        """Convert one dense ChunkRecord into a VectorRecord with deterministic id."""
        if not isinstance(record, ChunkRecord):
            raise VectorUpserterError("VectorUpserter expects ChunkRecord objects")
        if not record.dense_vector:
            raise VectorUpserterError(f"record {record.id} missing dense_vector")

        metadata = copy.deepcopy(record.metadata)
        stable_id = self.generate_chunk_id(record)
        metadata["original_chunk_id"] = record.id
        metadata["content_hash"] = _content_hash(record.text)
        return VectorRecord(
            id=stable_id,
            vector=[float(value) for value in record.dense_vector],
            text=record.text,
            metadata=metadata,
        )

    def generate_chunk_id(self, record: ChunkRecord) -> str:
        """Generate deterministic id from source path, chunk index, and content hash."""
        source_path = record.metadata.get("source_path")
        chunk_index = record.metadata.get("chunk_index")
        if not isinstance(source_path, str) or not source_path:
            raise VectorUpserterError(f"record {record.id} metadata.source_path is required")
        if chunk_index is None:
            raise VectorUpserterError(f"record {record.id} metadata.chunk_index is required")
        seed = f"{source_path}:{chunk_index}:{_content_hash(record.text)[:8]}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _content_hash(text: str) -> str:
    if not isinstance(text, str) or not text:
        raise VectorUpserterError("record text must be a non-empty string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_trace(trace: Any | None, name: str, data: dict[str, Any]) -> None:
    if hasattr(trace, "record_stage"):
        trace.record_stage(name, data)
