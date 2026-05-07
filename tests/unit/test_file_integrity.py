"""Tests for file integrity and SQLite ingestion history."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from libs.loader import DEFAULT_INTEGRITY_DB_PATH, FileIntegrityError, SQLiteIntegrityChecker


def test_compute_sha256_is_stable_for_same_file(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("same content", encoding="utf-8")
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")

    first = checker.compute_sha256(sample)
    second = checker.compute_sha256(sample)

    assert first == second
    assert len(first) == 64


def test_compute_sha256_missing_file_has_readable_error(tmp_path: Path) -> None:
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")

    with pytest.raises(FileIntegrityError, match="File not found"):
        checker.compute_sha256(tmp_path / "missing.txt")


def test_mark_success_makes_should_skip_true_and_stores_metadata(tmp_path: Path) -> None:
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")
    file_hash = "a" * 64

    assert checker.should_skip(file_hash) is False

    checker.mark_success(file_hash, "docs/a.pdf", file_size=123, chunk_count=7)

    assert checker.should_skip(file_hash) is True
    assert checker.get_record(file_hash) == {
        "file_hash": file_hash,
        "file_path": "docs/a.pdf",
        "file_size": 123,
        "status": "success",
        "processed_at": checker.get_record(file_hash)["processed_at"],
        "error_msg": None,
        "chunk_count": 7,
    }


def test_mark_failed_does_not_skip_and_stores_error(tmp_path: Path) -> None:
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")
    file_hash = "b" * 64

    checker.mark_failed(file_hash, "parse failed", file_path="docs/b.pdf")

    record = checker.get_record(file_hash)
    assert checker.should_skip(file_hash) is False
    assert record is not None
    assert record["status"] == "failed"
    assert record["error_msg"] == "parse failed"
    assert record["file_path"] == "docs/b.pdf"


def test_mark_success_overwrites_previous_failure(tmp_path: Path) -> None:
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")
    file_hash = "c" * 64

    checker.mark_failed(file_hash, "temporary failure", file_path="docs/c.pdf")
    checker.mark_success(file_hash, "docs/c.pdf", file_size=99, chunk_count=2)

    record = checker.get_record(file_hash)
    assert checker.should_skip(file_hash) is True
    assert record is not None
    assert record["status"] == "success"
    assert record["error_msg"] is None
    assert record["chunk_count"] == 2


def test_database_file_is_created_at_default_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    checker = SQLiteIntegrityChecker()

    assert checker.db_path == DEFAULT_INTEGRITY_DB_PATH
    assert (tmp_path / DEFAULT_INTEGRITY_DB_PATH).is_file()


def test_sqlite_uses_wal_mode(tmp_path: Path) -> None:
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")

    with checker._connect() as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]

    assert journal_mode.lower() == "wal"


def test_concurrent_success_writes_are_supported(tmp_path: Path) -> None:
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")

    def write(index: int) -> None:
        file_hash = f"{index:064x}"
        checker.mark_success(file_hash, f"docs/{index}.pdf", file_size=index, chunk_count=index + 1)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(write, range(16)))

    for index in range(16):
        file_hash = f"{index:064x}"
        assert checker.should_skip(file_hash) is True
        assert checker.get_record(file_hash)["chunk_count"] == index + 1


def test_invalid_hash_and_error_message_are_rejected(tmp_path: Path) -> None:
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")

    with pytest.raises(FileIntegrityError, match="file_hash"):
        checker.should_skip("")
    with pytest.raises(FileIntegrityError, match="error_msg"):
        checker.mark_failed("d" * 64, "")
