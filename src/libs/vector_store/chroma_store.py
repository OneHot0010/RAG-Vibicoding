"""Local persistent vector store with a Chroma-compatible role in the architecture."""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Any, Mapping

from core.settings import VectorStoreSettings
from libs.vector_store.base_vector_store import BaseVectorStore, VectorQueryResult, VectorRecord


class ChromaStoreError(ValueError):
    """Raised when ChromaStore receives invalid data or configuration."""


class ChromaStore(BaseVectorStore):
    """Minimal persistent vector store used as the default Chroma backend.

    The implementation is intentionally dependency-light for the early project
    phases. It preserves the same BaseVectorStore contract that a chromadb
    backed implementation will use later.
    """

    file_name = "records.json"

    def __init__(self, settings: VectorStoreSettings) -> None:
        if not settings.persist_path:
            raise ChromaStoreError("vector_store.persist_path is required")
        self.settings = settings
        self.persist_path = Path(settings.persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.records_path = self.persist_path / self.file_name
        self._records = self._load_records()

    def upsert(self, records: list[VectorRecord], trace: Any | None = None) -> None:
        if not isinstance(records, list):
            raise ChromaStoreError("records must be a list")
        for record in records:
            _validate_record(record)
            self._records[record.id] = record
        self._persist()

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[VectorQueryResult]:
        _validate_vector(vector, "query vector")
        if top_k <= 0:
            raise ChromaStoreError("top_k must be greater than 0")
        filters = filters or {}

        results: list[VectorQueryResult] = []
        for record in self._records.values():
            if not _metadata_matches(record.metadata, filters):
                continue
            results.append(
                VectorQueryResult(
                    id=record.id,
                    score=_cosine_similarity(vector, record.vector),
                    text=record.text,
                    metadata=dict(record.metadata),
                )
            )

        return sorted(results, key=lambda result: (-result.score, result.id))[:top_k]

    def get_by_ids(self, ids: list[str], trace: Any | None = None) -> list[VectorRecord]:
        """Return records by id in caller-provided order, omitting missing ids.

        The ingestion vector upserter stores deterministic vector ids while
        preserving source chunk ids in metadata.original_chunk_id, so this
        method accepts either identifier.
        """
        if not isinstance(ids, list) or not all(isinstance(item, str) and item for item in ids):
            raise ChromaStoreError("ids must be a list of non-empty strings")
        original_id_lookup = {
            str(record.metadata["original_chunk_id"]): record
            for record in self._records.values()
            if record.metadata.get("original_chunk_id")
        }
        return [
            self._records[record_id] if record_id in self._records else original_id_lookup[record_id]
            for record_id in ids
            if record_id in self._records or record_id in original_id_lookup
        ]

    def _load_records(self) -> dict[str, VectorRecord]:
        if not self.records_path.exists():
            return {}
        try:
            raw = json.loads(self.records_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ChromaStoreError(f"Invalid vector store file: {self.records_path}") from exc
        if not isinstance(raw, list):
            raise ChromaStoreError("Vector store file must contain a list")

        records: dict[str, VectorRecord] = {}
        for item in raw:
            if not isinstance(item, dict):
                raise ChromaStoreError("Vector store record must be an object")
            record = VectorRecord(
                id=str(item["id"]),
                vector=[float(value) for value in item["vector"]],
                text=str(item["text"]),
                metadata=dict(item.get("metadata") or {}),
            )
            _validate_record(record)
            records[record.id] = record
        return records

    def _persist(self) -> None:
        payload = [
            {
                "id": record.id,
                "vector": record.vector,
                "text": record.text,
                "metadata": record.metadata,
            }
            for record in sorted(self._records.values(), key=lambda item: item.id)
        ]
        self.records_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_record(record: VectorRecord) -> None:
    if not isinstance(record, VectorRecord):
        raise ChromaStoreError("record must be a VectorRecord")
    if not record.id:
        raise ChromaStoreError("record.id is required")
    if not record.text:
        raise ChromaStoreError("record.text is required")
    _validate_vector(record.vector, "record.vector")


def _validate_vector(vector: list[float], field_name: str) -> None:
    if not isinstance(vector, list) or not vector:
        raise ChromaStoreError(f"{field_name} must be a non-empty list")
    if not all(isinstance(value, (int, float)) for value in vector):
        raise ChromaStoreError(f"{field_name} must contain only numbers")


def _metadata_matches(metadata: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    for key, expected in filters.items():
        if metadata.get(key) != expected:
            return False
    return True


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ChromaStoreError("query vector dimension must match stored vector dimension")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
