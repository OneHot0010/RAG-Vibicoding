"""File integrity checks and SQLite-backed ingestion history."""

from __future__ import annotations

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


DEFAULT_INTEGRITY_DB_PATH = Path("data/db/ingestion_history.db")


class FileIntegrityError(RuntimeError):
    """Raised when file integrity state cannot be read or written."""


class FileIntegrityChecker(ABC):
    """Abstract interface for incremental ingestion integrity state."""

    @abstractmethod
    def compute_sha256(self, path: str | Path) -> str:
        """Return the SHA256 hex digest for a file."""

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """Return True when a file hash has already completed successfully."""

    @abstractmethod
    def mark_success(
        self,
        file_hash: str,
        file_path: str | Path,
        file_size: int | None = None,
        chunk_count: int | None = None,
    ) -> None:
        """Record a successful ingestion result."""

    @abstractmethod
    def mark_failed(self, file_hash: str, error_msg: str, file_path: str | Path | None = None) -> None:
        """Record a failed ingestion result."""


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """SQLite implementation of incremental ingestion history."""

    def __init__(self, db_path: str | Path = DEFAULT_INTEGRITY_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def compute_sha256(self, path: str | Path) -> str:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileIntegrityError(f"File not found: {file_path}")

        digest = hashlib.sha256()
        with file_path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        normalized_hash = _validate_hash(file_hash)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM ingestion_history WHERE file_hash = ? AND status = 'success'",
                (normalized_hash,),
            ).fetchone()
        return row is not None

    def mark_success(
        self,
        file_hash: str,
        file_path: str | Path,
        file_size: int | None = None,
        chunk_count: int | None = None,
    ) -> None:
        normalized_hash = _validate_hash(file_hash)
        normalized_path = _normalize_path(file_path)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_history (
                    file_hash, file_path, file_size, status, processed_at, error_msg, chunk_count
                )
                VALUES (?, ?, ?, 'success', CURRENT_TIMESTAMP, NULL, ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path = excluded.file_path,
                    file_size = excluded.file_size,
                    status = 'success',
                    processed_at = CURRENT_TIMESTAMP,
                    error_msg = NULL,
                    chunk_count = excluded.chunk_count
                """,
                (normalized_hash, normalized_path, file_size, chunk_count),
            )

    def mark_failed(self, file_hash: str, error_msg: str, file_path: str | Path | None = None) -> None:
        normalized_hash = _validate_hash(file_hash)
        if not isinstance(error_msg, str) or not error_msg.strip():
            raise FileIntegrityError("error_msg must be a non-empty string")
        normalized_path = _normalize_path(file_path) if file_path is not None else ""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_history (
                    file_hash, file_path, file_size, status, processed_at, error_msg, chunk_count
                )
                VALUES (?, ?, NULL, 'failed', CURRENT_TIMESTAMP, ?, NULL)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path = excluded.file_path,
                    status = 'failed',
                    processed_at = CURRENT_TIMESTAMP,
                    error_msg = excluded.error_msg,
                    chunk_count = NULL
                """,
                (normalized_hash, normalized_path, error_msg),
            )

    def get_record(self, file_hash: str) -> dict[str, Any] | None:
        """Return a stored ingestion history row by hash."""
        normalized_hash = _validate_hash(file_hash)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT file_hash, file_path, file_size, status, processed_at, error_msg, chunk_count
                FROM ingestion_history
                WHERE file_hash = ?
                """,
                (normalized_hash,),
            ).fetchone()
        return dict(row) if row is not None else None

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    status TEXT NOT NULL CHECK(status IN ('success', 'failed', 'processing')),
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_msg TEXT,
                    chunk_count INTEGER
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_status ON ingestion_history(status)")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_processed_at ON ingestion_history(processed_at)"
            )

    def _connect(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(self.db_path, timeout=5.0)
        except sqlite3.Error as exc:
            raise FileIntegrityError(f"Failed to open ingestion history database: {self.db_path}") from exc
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        return connection


def _validate_hash(file_hash: str) -> str:
    if not isinstance(file_hash, str) or not file_hash.strip():
        raise FileIntegrityError("file_hash must be a non-empty string")
    return file_hash.strip()


def _normalize_path(path: str | Path) -> str:
    normalized = str(path)
    if not normalized:
        raise FileIntegrityError("file_path must be a non-empty path")
    return normalized
