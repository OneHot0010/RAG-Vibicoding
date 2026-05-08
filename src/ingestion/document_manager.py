"""Document lifecycle management across local RAG stores."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.types import ChunkRecord
from ingestion.storage import BM25Indexer, ImageStorage


@dataclass(frozen=True)
class DocumentInfo:
    """Summary information for one ingested document."""

    source_path: str
    collection: str | None
    doc_hash: str | None
    chunk_count: int
    image_count: int
    processed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize document information."""
        return {
            "source_path": self.source_path,
            "collection": self.collection,
            "doc_hash": self.doc_hash,
            "chunk_count": self.chunk_count,
            "image_count": self.image_count,
            "processed_at": self.processed_at,
        }


@dataclass(frozen=True)
class DocumentDetail:
    """Detailed document data for Dashboard inspection."""

    document: DocumentInfo
    chunks: list[dict[str, Any]]
    images: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize document detail."""
        return {
            "document": self.document.to_dict(),
            "chunks": self.chunks,
            "images": self.images,
        }


@dataclass(frozen=True)
class DeleteResult:
    """Counts returned after coordinated document deletion."""

    source_path: str
    collection: str | None
    vector_deleted: int
    bm25_deleted: int
    image_deleted: int
    integrity_deleted: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize deletion counts."""
        return {
            "source_path": self.source_path,
            "collection": self.collection,
            "vector_deleted": self.vector_deleted,
            "bm25_deleted": self.bm25_deleted,
            "image_deleted": self.image_deleted,
            "integrity_deleted": self.integrity_deleted,
        }


@dataclass(frozen=True)
class CollectionStats:
    """Aggregated collection statistics."""

    collection: str | None
    document_count: int
    chunk_count: int
    image_count: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize collection statistics."""
        return {
            "collection": self.collection,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "image_count": self.image_count,
        }


class DocumentManager:
    """Coordinate document list/detail/delete operations across local stores."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.chroma_records_path = self.data_dir / "db" / "chroma" / "records.json"
        self.bm25_dir = self.data_dir / "db" / "bm25"
        self.image_storage = ImageStorage(
            images_root=self.data_dir / "images",
            db_path=self.data_dir / "db" / "image_index.db",
        )
        self.integrity_db_path = self.data_dir / "db" / "ingestion_history.db"

    def list_documents(self, collection: str | None = None) -> list[DocumentInfo]:
        """Return ingested documents grouped by source path and collection."""
        documents: dict[tuple[str, str | None], dict[str, Any]] = {}
        for record in self._load_vector_records():
            metadata = _metadata(record)
            source = _source_path(metadata)
            if source is None:
                continue
            record_collection = _collection(metadata)
            if collection is not None and record_collection != collection:
                continue
            key = (source, record_collection)
            item = documents.setdefault(
                key,
                {"source_path": source, "collection": record_collection, "doc_hash": _doc_hash(metadata), "chunk_count": 0},
            )
            item["chunk_count"] += 1
            item["doc_hash"] = item["doc_hash"] or _doc_hash(metadata)

        history = self._history_by_path()
        infos = []
        for item in documents.values():
            doc_hash = item["doc_hash"]
            history_row = history.get(item["source_path"]) or (self._history_by_hash().get(doc_hash) if doc_hash else None)
            image_count = len(self.image_storage.list_images(collection=item["collection"], doc_hash=doc_hash)) if doc_hash else 0
            infos.append(
                DocumentInfo(
                    source_path=item["source_path"],
                    collection=item["collection"],
                    doc_hash=doc_hash,
                    chunk_count=item["chunk_count"],
                    image_count=image_count,
                    processed_at=history_row.get("processed_at") if history_row else None,
                )
            )
        return sorted(infos, key=lambda item: (item.collection or "", item.source_path))

    def get_document_detail(self, doc_id: str) -> DocumentDetail:
        """Return chunks and images for a document id/source/hash."""
        records = [record for record in self._load_vector_records() if _matches_doc_id(record, doc_id)]
        if not records:
            raise ValueError(f"document not found: {doc_id}")
        metadata = _metadata(records[0])
        source = _source_path(metadata) or doc_id
        collection = _collection(metadata)
        doc_hash = _doc_hash(metadata)
        info = next(
            (
                document
                for document in self.list_documents(collection=collection)
                if document.source_path == source and document.doc_hash == doc_hash
            ),
            DocumentInfo(source, collection, doc_hash, len(records), 0),
        )
        images = [image.to_dict() for image in self.image_storage.list_images(collection=collection, doc_hash=doc_hash)] if doc_hash else []
        chunks = [
            {
                "id": str(record.get("id") or ""),
                "text": str(record.get("text") or ""),
                "metadata": _metadata(record),
            }
            for record in sorted(records, key=_chunk_sort_key)
        ]
        return DocumentDetail(document=info, chunks=chunks, images=images)

    def delete_document(self, source_path: str, collection: str | None = None) -> DeleteResult:
        """Delete a document from vector, BM25, image, and integrity stores."""
        if not isinstance(source_path, str) or not source_path.strip():
            raise ValueError("source_path must be a non-empty string")
        normalized_source = source_path.strip()
        matching_records = [
            record
            for record in self._load_vector_records()
            if _source_path(_metadata(record)) == normalized_source
            and (collection is None or _collection(_metadata(record)) == collection)
        ]
        doc_hashes = {_doc_hash(_metadata(record)) for record in matching_records if _doc_hash(_metadata(record))}
        vector_deleted = self._delete_vector_records(normalized_source, collection)
        bm25_deleted = self._delete_bm25_records(normalized_source, collection)
        image_deleted = sum(
            self.image_storage.delete_images(collection=collection, doc_hash=doc_hash)
            for doc_hash in sorted(doc_hashes)
        )
        integrity_deleted = self._delete_integrity_rows(normalized_source, doc_hashes)
        return DeleteResult(
            source_path=normalized_source,
            collection=collection,
            vector_deleted=vector_deleted,
            bm25_deleted=bm25_deleted,
            image_deleted=image_deleted,
            integrity_deleted=integrity_deleted,
        )

    def get_collection_stats(self, collection: str | None = None) -> CollectionStats:
        """Return aggregate statistics for all or one collection."""
        documents = self.list_documents(collection=collection)
        return CollectionStats(
            collection=collection,
            document_count=len(documents),
            chunk_count=sum(document.chunk_count for document in documents),
            image_count=sum(document.image_count for document in documents),
        )

    def _load_vector_records(self) -> list[dict[str, Any]]:
        if not self.chroma_records_path.is_file():
            return []
        try:
            payload = json.loads(self.chroma_records_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return [record for record in payload if isinstance(record, dict)] if isinstance(payload, list) else []

    def _write_vector_records(self, records: list[dict[str, Any]]) -> None:
        self.chroma_records_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_records_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    def _delete_vector_records(self, source_path: str, collection: str | None) -> int:
        records = self._load_vector_records()
        kept = [
            record
            for record in records
            if not (
                _source_path(_metadata(record)) == source_path
                and (collection is None or _collection(_metadata(record)) == collection)
            )
        ]
        deleted = len(records) - len(kept)
        if deleted:
            self._write_vector_records(kept)
        return deleted

    def _delete_bm25_records(self, source_path: str, collection: str | None) -> int:
        try:
            indexer = BM25Indexer.load(self.bm25_dir)
        except Exception:
            return 0
        matching_ids = [
            record_id
            for record_id, record in indexer.records.items()
            if record.metadata.get("source_path") == source_path
            and (collection is None or record.metadata.get("collection") == collection)
        ]
        for record_id in matching_ids:
            indexer.records.pop(record_id, None)
        if matching_ids:
            indexer._rebuild_index()
            indexer.save()
        return len(matching_ids)

    def _delete_integrity_rows(self, source_path: str, doc_hashes: set[str]) -> int:
        if not self.integrity_db_path.is_file():
            return 0
        clauses = ["file_path = ?"]
        params: list[str] = [source_path]
        for doc_hash in sorted(doc_hashes):
            clauses.append("file_hash = ?")
            params.append(doc_hash)
        with sqlite3.connect(self.integrity_db_path) as connection:
            cursor = connection.execute(
                f"DELETE FROM ingestion_history WHERE {' OR '.join(clauses)}",
                params,
            )
        return int(cursor.rowcount)

    def _history_by_path(self) -> dict[str, dict[str, Any]]:
        rows = self._history_rows()
        return {row["file_path"]: row for row in rows if row.get("file_path")}

    def _history_by_hash(self) -> dict[str, dict[str, Any]]:
        rows = self._history_rows()
        return {row["file_hash"]: row for row in rows if row.get("file_hash")}

    def _history_rows(self) -> list[dict[str, Any]]:
        if not self.integrity_db_path.is_file():
            return []
        try:
            with sqlite3.connect(self.integrity_db_path) as connection:
                connection.row_factory = sqlite3.Row
                rows = connection.execute(
                    "SELECT file_hash, file_path, status, processed_at, chunk_count FROM ingestion_history"
                ).fetchall()
        except sqlite3.Error:
            return []
        return [dict(row) for row in rows]


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _source_path(metadata: dict[str, Any]) -> str | None:
    source = metadata.get("source_path") or metadata.get("source")
    return str(source) if source else None


def _collection(metadata: dict[str, Any]) -> str | None:
    collection = metadata.get("collection")
    return str(collection) if collection else None


def _doc_hash(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("file_hash") or metadata.get("doc_hash")
    return str(value) if value else None


def _matches_doc_id(record: dict[str, Any], doc_id: str) -> bool:
    metadata = _metadata(record)
    source = _source_path(metadata)
    candidates = {
        str(record.get("id") or ""),
        str(metadata.get("original_chunk_id") or ""),
        str(metadata.get("file_hash") or ""),
        str(metadata.get("doc_hash") or ""),
    }
    if source:
        source_path = Path(source)
        candidates.update({source, source_path.name, source_path.stem})
    return doc_id in candidates


def _chunk_sort_key(record: dict[str, Any]) -> tuple[int, str]:
    metadata = _metadata(record)
    try:
        index = int(metadata.get("chunk_index"))
    except (TypeError, ValueError):
        index = 0
    return index, str(record.get("id") or "")
