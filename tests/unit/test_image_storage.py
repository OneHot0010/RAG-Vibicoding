"""Tests for SQLite-indexed image storage."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from core.trace.trace_context import TraceContext
from ingestion.storage import DEFAULT_IMAGE_DB_PATH, DEFAULT_IMAGE_ROOT, ImageRecord, ImageStorage, ImageStorageError


def make_storage(tmp_path: Path) -> ImageStorage:
    return ImageStorage(images_root=tmp_path / "images", db_path=tmp_path / "image_index.db")


def test_save_image_writes_file_and_index_record(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)

    record = storage.save_image("img-1", b"png-bytes", collection="docs", doc_hash="a" * 64, page_num=3)

    assert isinstance(record, ImageRecord)
    assert record.image_id == "img-1"
    assert record.collection == "docs"
    assert record.doc_hash == "a" * 64
    assert record.page_num == 3
    assert Path(record.file_path).read_bytes() == b"png-bytes"
    assert Path(record.file_path).parent == tmp_path / "images" / "docs"
    assert storage.get_image_path("img-1") == Path(record.file_path)


def test_index_mapping_is_persistent(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)
    saved = storage.save_image("img-2", b"data", collection="docs", doc_hash="doc-a")

    loaded = ImageStorage(images_root=tmp_path / "images", db_path=tmp_path / "image_index.db")

    assert loaded.get_record("img-2").to_dict() == saved.to_dict()


def test_upsert_same_image_id_replaces_file_and_metadata(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)
    first = storage.save_image("img-3", b"old", collection="docs", doc_hash="doc-a")
    second = storage.save_image("img-3", b"new", collection="notes", doc_hash="doc-b", page_num=2)

    assert first.file_path != second.file_path
    assert storage.get_record("img-3").collection == "notes"
    assert Path(second.file_path).read_bytes() == b"new"
    assert len(storage.list_images()) == 1


def test_list_images_filters_by_collection_and_doc_hash(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)
    storage.save_image("a", b"a", collection="docs", doc_hash="doc-a")
    storage.save_image("b", b"b", collection="docs", doc_hash="doc-b")
    storage.save_image("c", b"c", collection="notes", doc_hash="doc-a")

    assert [record.image_id for record in storage.list_images(collection="docs")] == ["a", "b"]
    assert [record.image_id for record in storage.list_images(doc_hash="doc-a")] == ["a", "c"]
    assert [record.image_id for record in storage.list_images(collection="docs", doc_hash="doc-b")] == ["b"]


def test_save_file_copies_existing_image_with_source_suffix(tmp_path: Path) -> None:
    source = tmp_path / "source.jpeg"
    source.write_bytes(b"jpeg")
    storage = make_storage(tmp_path)

    record = storage.save_file("photo", source, collection="docs")

    assert Path(record.file_path).suffix == ".jpeg"
    assert Path(record.file_path).read_bytes() == b"jpeg"


def test_delete_images_removes_files_and_rows(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)
    keep = storage.save_image("keep", b"keep", collection="docs", doc_hash="doc-a")
    delete = storage.save_image("delete", b"delete", collection="docs", doc_hash="doc-b")

    count = storage.delete_images(collection="docs", doc_hash="doc-b")

    assert count == 1
    assert Path(keep.file_path).is_file()
    assert not Path(delete.file_path).exists()
    assert storage.get_record("delete") is None


def test_sqlite_uses_wal_mode_and_expected_indexes(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)

    with storage._connect() as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(image_index)").fetchall()}

    assert journal_mode.lower() == "wal"
    assert {"idx_collection", "idx_doc_hash"}.issubset(indexes)


def test_default_paths_are_used_relative_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    storage = ImageStorage()

    assert storage.images_root == DEFAULT_IMAGE_ROOT
    assert storage.db_path == DEFAULT_IMAGE_DB_PATH
    assert (tmp_path / DEFAULT_IMAGE_DB_PATH).is_file()
    assert (tmp_path / DEFAULT_IMAGE_ROOT).is_dir()


def test_concurrent_writes_are_supported(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)

    def save(index: int) -> None:
        storage.save_image(f"img-{index}", f"bytes-{index}".encode("utf-8"), collection="docs", doc_hash="doc")

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(save, range(16)))

    assert len(storage.list_images(collection="docs")) == 16
    assert storage.get_image_path("img-7").read_bytes() == b"bytes-7"


def test_trace_records_save_stage(tmp_path: Path) -> None:
    trace = TraceContext()

    make_storage(tmp_path).save_image("trace-img", b"bytes", collection="docs", doc_hash="doc", trace=trace)

    assert trace.stages == [
        {
            "name": "image_storage.save_image",
            "data": {"image_id": "trace-img", "collection": "docs", "doc_hash": "doc"},
        }
    ]


def test_invalid_inputs_have_readable_errors(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)

    with pytest.raises(ImageStorageError, match="image_id"):
        storage.save_image("", b"bytes")
    with pytest.raises(ImageStorageError, match="image_bytes"):
        storage.save_image("img", b"")
    with pytest.raises(ImageStorageError, match="page_num"):
        storage.save_image("img", b"bytes", page_num=-1)
    with pytest.raises(ImageStorageError, match="extension"):
        storage.save_image("img", b"bytes", extension="../png")
    with pytest.raises(ImageStorageError, match="not found"):
        storage.save_file("missing", tmp_path / "missing.png")


def test_corrupt_database_error_is_readable(tmp_path: Path) -> None:
    db_path = tmp_path / "not_a_db"
    db_path.mkdir()

    with pytest.raises((ImageStorageError, sqlite3.Error)):
        ImageStorage(images_root=tmp_path / "images", db_path=db_path)
