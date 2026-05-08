"""Sparse keyword retrieval backed by BM25 and vector-store records."""

from __future__ import annotations

from typing import Any

from core.settings import Settings
from core.types import RetrievalResult
from ingestion.storage import BM25Hit, BM25Indexer
from libs.vector_store import BaseVectorStore, VectorRecord, VectorStoreFactory


class SparseRetrieverError(ValueError):
    """Raised when sparse retrieval input or backend output is invalid."""


class SparseRetriever:
    """Query BM25 and hydrate hits from vector storage."""

    def __init__(
        self,
        settings: Settings,
        bm25_indexer: BM25Indexer | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.bm25_indexer = bm25_indexer or BM25Indexer.load()
        self.vector_store = vector_store or VectorStoreFactory.create(settings)

    def retrieve(
        self,
        keywords: list[str],
        top_k: int,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        """Return BM25-ranked retrieval results with hydrated text and metadata."""
        normalized_keywords = _normalize_keywords(keywords)
        if top_k <= 0:
            raise SparseRetrieverError("top_k must be greater than 0")

        hits = self.bm25_indexer.query(normalized_keywords, top_k=top_k)
        _validate_hits(hits)
        if not hits:
            _record_trace(trace, "sparse_retriever.retrieve", {"top_k": top_k, "result_count": 0})
            return []

        records = self.vector_store.get_by_ids([hit.chunk_id for hit in hits], trace=trace)
        _validate_records(records)
        records_by_id = _records_by_retrieval_id(records)
        results = [
            RetrievalResult(
                chunk_id=hit.chunk_id,
                score=float(hit.score),
                text=records_by_id[hit.chunk_id].text,
                metadata=dict(records_by_id[hit.chunk_id].metadata),
            )
            for hit in hits
            if hit.chunk_id in records_by_id
        ]
        _record_trace(
            trace,
            "sparse_retriever.retrieve",
            {"top_k": top_k, "result_count": len(results), "keyword_count": len(normalized_keywords)},
        )
        return results


def _normalize_keywords(keywords: list[str]) -> list[str]:
    if not isinstance(keywords, list) or not all(isinstance(keyword, str) for keyword in keywords):
        raise SparseRetrieverError("keywords must be a list[str]")
    normalized = [keyword.strip().lower() for keyword in keywords if keyword.strip()]
    if not normalized:
        raise SparseRetrieverError("keywords must contain at least one non-empty term")
    return normalized


def _validate_hits(hits: list[BM25Hit]) -> None:
    if not isinstance(hits, list):
        raise SparseRetrieverError("BM25 indexer must return a list")
    for hit in hits:
        if not isinstance(hit, BM25Hit):
            raise SparseRetrieverError("BM25 indexer must return BM25Hit objects")


def _validate_records(records: list[VectorRecord]) -> None:
    if not isinstance(records, list):
        raise SparseRetrieverError("vector store get_by_ids must return a list")
    for record in records:
        if not isinstance(record, VectorRecord):
            raise SparseRetrieverError("vector store get_by_ids must return VectorRecord objects")


def _records_by_retrieval_id(records: list[VectorRecord]) -> dict[str, VectorRecord]:
    records_by_id = {record.id: record for record in records}
    for record in records:
        original_chunk_id = record.metadata.get("original_chunk_id")
        if isinstance(original_chunk_id, str) and original_chunk_id:
            records_by_id[original_chunk_id] = record
    return records_by_id


def _record_trace(trace: Any | None, name: str, data: dict[str, Any]) -> None:
    if hasattr(trace, "record_stage"):
        trace.record_stage(name, data)
