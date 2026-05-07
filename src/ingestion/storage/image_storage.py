"""SQLite-indexed image file storage for ingestion outputs."""

from __future__ import annotations

import re
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_IMAGE_ROOT = Path("data/images")
DEFAULT_IMAGE_DB_PATH = Path("data/db/image_index.db")


class ImageStorageError(ValueError):
    """Raised when image storage input or persistence fails."""


@dataclass(frozen=True)
class ImageRecord:
    """Stored image index row."""

    image_id: str
    file_path: str
    collection: str | None = None
    doc_hash: str | None = None
    page_num: int | None = None
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the image record for callers and tests."""
        return {
            "image_id": self.image_id,
            "file_path": self.file_path,
            "collection": self.collection,
            "doc_hash": self.doc_hash,
            "page_num": self.page_num,
            "created_at": self.created_at,
        }


class ImageStorage:
    """Store image bytes on disk and maintain an image_id to path SQLite index."""

    def __init__(
        self,
        images_root: str | Path = DEFAULT_IMAGE_ROOT,
        db_path: str | Path = DEFAULT_IMAGE_DB_PATH,
    ) -> None:
        self.images_root = Path(images_root)
        self.db_path = Path(db_path)
        self.images_root.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def save_image(
        self,
        image_id: str,
        image_bytes: bytes,
        collection: str = "default",
        doc_hash: str | None = None,
        page_num: int | None = None,
        extension: str = ".png",
        trace: Any | None = None,
    ) -> ImageRecord:
        """Persist image bytes and upsert the SQLite mapping."""
        normalized_id = _validate_non_empty("image_id", image_id)
        if not isinstance(image_bytes, bytes) or not image_bytes:
            raise ImageStorageError("image_bytes must be non-empty bytes")
        normalized_collection = _validate_non_empty("collection", collection)
        normalized_doc_hash = _optional_non_empty("doc_hash", doc_hash)
        normalized_page_num = _normalize_page_num(page_num)
        suffix = _normalize_extension(extension)

        collection_dir = self.images_root / _safe_path_part(normalized_collection)
        collection_dir.mkdir(parents=True, exist_ok=True)
        file_path = collection_dir / f"{_safe_path_part(normalized_id)}{suffix}"
        file_path.write_bytes(image_bytes)

        record = self._upsert_record(
            image_id=normalized_id,
            file_path=str(file_path),
            collection=normalized_collection,
            doc_hash=normalized_doc_hash,
            page_num=normalized_page_num,
        )
        _record_trace(
            trace,
            "image_storage.save_image",
            {"image_id": normalized_id, "collection": normalized_collection, "doc_hash": normalized_doc_hash},
        )
        return record

    def save_file(
        self,
        image_id: str,
        source_path: str | Path,
        collection: str = "default",
        doc_hash: str | None = None,
        page_num: int | None = None,
        trace: Any | None = None,
    ) -> ImageRecord:
        """Copy an existing image file into managed storage."""
        path = Path(source_path)
        if not path.is_file():
            raise ImageStorageError(f"image file not found: {path}")
        record = self.save_image(
            image_id=image_id,
            image_bytes=path.read_bytes(),
            collection=collection,
            doc_hash=doc_hash,
            page_num=page_num,
            extension=path.suffix or ".png",
            trace=trace,
        )
        return record

    def get_record(self, image_id: str) -> ImageRecord | None:
        """Return an image index record by id."""
        normalized_id = _validate_non_empty("image_id", image_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT image_id, file_path, collection, doc_hash, page_num, created_at
                FROM image_index
                WHERE image_id = ?
                """,
                (normalized_id,),
            ).fetchone()
        return _record_from_row(row) if row is not None else None

    def get_image_path(self, image_id: str) -> Path | None:
        """Return the stored file path for an image id."""
        record = self.get_record(image_id)
        return Path(record.file_path) if record is not None else None

    def list_images(self, collection: str | None = None, doc_hash: str | None = None) -> list[ImageRecord]:
        """List image records, optionally filtered by collection and/or document hash."""
        clauses: list[str] = []
        params: list[str] = []
        if collection is not None:
            clauses.append("collection = ?")
            params.append(_validate_non_empty("collection", collection))
        if doc_hash is not None:
            clauses.append("doc_hash = ?")
            params.append(_validate_non_empty("doc_hash", doc_hash))

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT image_id, file_path, collection, doc_hash, page_num, created_at
                FROM image_index
                {where}
                ORDER BY created_at, image_id
                """,
                tuple(params),
            ).fetchall()
        return [_record_from_row(row) for row in rows]

    def delete_images(self, collection: str | None = None, doc_hash: str | None = None) -> int:
        """Delete indexed image files matching filters and return the deleted row count."""
        records = self.list_images(collection=collection, doc_hash=doc_hash)
        if not records:
            return 0
        for record in records:
            path = Path(record.file_path)
            if path.exists():
                path.unlink()

        clauses: list[str] = []
        params: list[str] = []
        if collection is not None:
            clauses.append("collection = ?")
            params.append(_validate_non_empty("collection", collection))
        if doc_hash is not None:
            clauses.append("doc_hash = ?")
            params.append(_validate_non_empty("doc_hash", doc_hash))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            cursor = connection.execute(f"DELETE FROM image_index {where}", tuple(params))
        return int(cursor.rowcount)

    def _upsert_record(
        self,
        image_id: str,
        file_path: str,
        collection: str,
        doc_hash: str | None,
        page_num: int | None,
    ) -> ImageRecord:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO image_index (image_id, file_path, collection, doc_hash, page_num, created_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(image_id) DO UPDATE SET
                    file_path = excluded.file_path,
                    collection = excluded.collection,
                    doc_hash = excluded.doc_hash,
                    page_num = excluded.page_num,
                    created_at = CURRENT_TIMESTAMP
                """,
                (image_id, file_path, collection, doc_hash, page_num),
            )
        record = self.get_record(image_id)
        if record is None:
            raise ImageStorageError(f"failed to persist image record: {image_id}")
        return record

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS image_index (
                    image_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    collection TEXT,
                    doc_hash TEXT,
                    page_num INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_collection ON image_index(collection)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_doc_hash ON image_index(doc_hash)")

    def _connect(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(self.db_path, timeout=5.0)
        except sqlite3.Error as exc:
            raise ImageStorageError(f"Failed to open image index database: {self.db_path}") from exc
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        return connection


def _record_from_row(row: sqlite3.Row) -> ImageRecord:
    return ImageRecord(
        image_id=str(row["image_id"]),
        file_path=str(row["file_path"]),
        collection=row["collection"],
        doc_hash=row["doc_hash"],
        page_num=row["page_num"],
        created_at=row["created_at"],
    )


def _validate_non_empty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ImageStorageError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_non_empty(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    return _validate_non_empty(name, value)


def _normalize_page_num(page_num: int | None) -> int | None:
    if page_num is None:
        return None
    if isinstance(page_num, bool) or int(page_num) < 0:
        raise ImageStorageError("page_num must be non-negative when provided")
    return int(page_num)


def _normalize_extension(extension: str) -> str:
    suffix = _validate_non_empty("extension", extension)
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    if not re.fullmatch(r"\.[A-Za-z0-9]+", suffix):
        raise ImageStorageError("extension must contain only letters or numbers")
    return suffix.lower()


def _safe_path_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    if not safe:
        raise ImageStorageError("path component becomes empty after normalization")
    return safe


def _record_trace(trace: Any | None, name: str, data: dict[str, Any]) -> None:
    if hasattr(trace, "record_stage"):
        trace.record_stage(name, data)
