"""Tests for DocumentManager coordinated document lifecycle operations."""

from __future__ import annotations

import json
from pathlib import Path

from core.types import ChunkRecord
from ingestion.document_manager import DocumentManager
from ingestion.embedding.sparse_encoder import SparseEncodingResult
from ingestion.storage import BM25Indexer, ImageStorage
from libs.loader import SQLiteIntegrityChecker
from observability.dashboard.services.data_service import DataService


def seed_document(data_dir: Path) -> None:
    chroma = data_dir / "db" / "chroma"
    chroma.mkdir(parents=True)
    records = [
        {
            "id": "vec-a",
            "vector": [1.0, 0.0],
            "text": "alpha chunk",
            "metadata": {
                "source_path": "docs/a.pdf",
                "collection": "docs",
                "file_hash": "hash-a",
                "chunk_index": 0,
                "original_chunk_id": "chunk-a",
            },
        },
        {
            "id": "vec-b",
            "vector": [0.0, 1.0],
            "text": "beta chunk",
            "metadata": {
                "source_path": "docs/a.pdf",
                "collection": "docs",
                "file_hash": "hash-a",
                "chunk_index": 1,
                "original_chunk_id": "chunk-b",
            },
        },
        {
            "id": "vec-c",
            "vector": [0.5, 0.5],
            "text": "other chunk",
            "metadata": {
                "source_path": "docs/b.pdf",
                "collection": "docs",
                "file_hash": "hash-b",
                "chunk_index": 0,
                "original_chunk_id": "chunk-c",
            },
        },
    ]
    (chroma / "records.json").write_text(json.dumps(records), encoding="utf-8")

    sparse_records = [
        ChunkRecord(
            id="chunk-a",
            text="alpha chunk",
            metadata={"source_path": "docs/a.pdf", "collection": "docs", "doc_length": 2},
            sparse_vector={"alpha": 1.0, "chunk": 1.0},
        ),
        ChunkRecord(
            id="chunk-b",
            text="beta chunk",
            metadata={"source_path": "docs/a.pdf", "collection": "docs", "doc_length": 2},
            sparse_vector={"beta": 1.0, "chunk": 1.0},
        ),
        ChunkRecord(
            id="chunk-c",
            text="other chunk",
            metadata={"source_path": "docs/b.pdf", "collection": "docs", "doc_length": 2},
            sparse_vector={"other": 1.0, "chunk": 1.0},
        ),
    ]
    indexer = BM25Indexer(index_dir=data_dir / "db" / "bm25")
    indexer.build(SparseEncodingResult(sparse_records, {"alpha": 1, "beta": 1, "other": 1, "chunk": 3}, 3, 2.0))
    indexer.save()

    images = ImageStorage(images_root=data_dir / "images", db_path=data_dir / "db" / "image_index.db")
    images.save_image("img-a", b"image-a", collection="docs", doc_hash="hash-a")
    images.save_image("img-b", b"image-b", collection="docs", doc_hash="hash-b")

    history = SQLiteIntegrityChecker(data_dir / "db" / "ingestion_history.db")
    history.mark_success("hash-a", "docs/a.pdf", file_size=123, chunk_count=2)
    history.mark_success("hash-b", "docs/b.pdf", file_size=456, chunk_count=1)


def test_document_manager_lists_documents_with_counts(tmp_path: Path) -> None:
    seed_document(tmp_path)

    documents = DocumentManager(tmp_path).list_documents(collection="docs")

    assert [document.to_dict() for document in documents] == [
        {
            "source_path": "docs/a.pdf",
            "collection": "docs",
            "doc_hash": "hash-a",
            "chunk_count": 2,
            "image_count": 1,
            "processed_at": documents[0].processed_at,
        },
        {
            "source_path": "docs/b.pdf",
            "collection": "docs",
            "doc_hash": "hash-b",
            "chunk_count": 1,
            "image_count": 1,
            "processed_at": documents[1].processed_at,
        },
    ]
    assert documents[0].processed_at is not None


def test_document_manager_returns_detail_by_source_name(tmp_path: Path) -> None:
    seed_document(tmp_path)

    detail = DocumentManager(tmp_path).get_document_detail("a.pdf").to_dict()

    assert detail["document"]["source_path"] == "docs/a.pdf"
    assert [chunk["id"] for chunk in detail["chunks"]] == ["vec-a", "vec-b"]
    assert detail["chunks"][0]["metadata"]["original_chunk_id"] == "chunk-a"
    assert [image["image_id"] for image in detail["images"]] == ["img-a"]


def test_document_manager_collection_stats(tmp_path: Path) -> None:
    seed_document(tmp_path)

    stats = DocumentManager(tmp_path).get_collection_stats("docs").to_dict()

    assert stats == {"collection": "docs", "document_count": 2, "chunk_count": 3, "image_count": 2}


def test_document_manager_delete_document_updates_all_local_stores(tmp_path: Path) -> None:
    seed_document(tmp_path)
    manager = DocumentManager(tmp_path)

    result = manager.delete_document("docs/a.pdf", collection="docs").to_dict()

    assert result == {
        "source_path": "docs/a.pdf",
        "collection": "docs",
        "vector_deleted": 2,
        "bm25_deleted": 2,
        "image_deleted": 1,
        "integrity_deleted": 1,
    }
    assert [document.source_path for document in manager.list_documents()] == ["docs/b.pdf"]
    remaining_vectors = json.loads((tmp_path / "db" / "chroma" / "records.json").read_text(encoding="utf-8"))
    assert [record["id"] for record in remaining_vectors] == ["vec-c"]
    assert list(BM25Indexer.load(tmp_path / "db" / "bm25").records) == ["chunk-c"]
    assert ImageStorage(images_root=tmp_path / "images", db_path=tmp_path / "db" / "image_index.db").get_record("img-a") is None
    assert SQLiteIntegrityChecker(tmp_path / "db" / "ingestion_history.db").get_record("hash-a") is None


def test_document_manager_delete_respects_collection_filter(tmp_path: Path) -> None:
    seed_document(tmp_path)
    records_path = tmp_path / "db" / "chroma" / "records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    records.append(
        {
            "id": "vec-a-other",
            "vector": [1.0, 1.0],
            "text": "same source in other collection",
            "metadata": {
                "source_path": "docs/a.pdf",
                "collection": "other",
                "file_hash": "hash-a-other",
                "chunk_index": 0,
                "original_chunk_id": "chunk-a-other",
            },
        }
    )
    records_path.write_text(json.dumps(records), encoding="utf-8")

    result = DocumentManager(tmp_path).delete_document("docs/a.pdf", collection="docs").to_dict()

    assert result["vector_deleted"] == 2
    remaining_vectors = json.loads(records_path.read_text(encoding="utf-8"))
    assert [record["id"] for record in remaining_vectors] == ["vec-c", "vec-a-other"]


def test_document_manager_delete_missing_document_is_noop(tmp_path: Path) -> None:
    seed_document(tmp_path)

    result = DocumentManager(tmp_path).delete_document("docs/missing.pdf", collection="docs").to_dict()

    assert result == {
        "source_path": "docs/missing.pdf",
        "collection": "docs",
        "vector_deleted": 0,
        "bm25_deleted": 0,
        "image_deleted": 0,
        "integrity_deleted": 0,
    }


def test_document_manager_rejects_blank_delete_source_path(tmp_path: Path) -> None:
    try:
        DocumentManager(tmp_path).delete_document("   ")
    except ValueError as exc:
        assert "source_path must be a non-empty string" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_document_manager_missing_detail_has_readable_error(tmp_path: Path) -> None:
    try:
        DocumentManager(tmp_path).get_document_detail("missing")
    except ValueError as exc:
        assert "document not found" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_data_service_wraps_document_manager(tmp_path: Path) -> None:
    seed_document(tmp_path)
    service = DataService(data_dir=tmp_path)

    assert service.get_collection_stats("docs")["document_count"] == 2
    assert service.list_documents("docs")[0]["source_path"] == "docs/a.pdf"
    assert service.get_document_detail("hash-a")["document"]["chunk_count"] == 2
