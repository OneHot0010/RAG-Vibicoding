"""Tests for the get_document_summary MCP tool."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mcp_server.protocol_handler import ProtocolError
from mcp_server.tools.get_document_summary import get_document_summary, get_document_summary_tool_spec


def write_records(data_dir: Path) -> None:
    chroma = data_dir / "db" / "chroma"
    chroma.mkdir(parents=True)
    (chroma / "records.json").write_text(
        json.dumps(
            [
                {
                    "id": "vec-b",
                    "vector": [1.0],
                    "text": "Second Azure chunk",
                    "metadata": {
                        "source_path": "docs/azure.pdf",
                        "file_hash": "hash-azure",
                        "collection": "docs",
                        "chunk_index": 1,
                        "title": "Azure Guide",
                        "summary": "Covers deployments and versions.",
                        "tags": ["azure", "deployments"],
                        "page": 2,
                    },
                },
                {
                    "id": "vec-a",
                    "vector": [1.0],
                    "text": "First Azure chunk",
                    "metadata": {
                        "source_path": "docs/azure.pdf",
                        "file_hash": "hash-azure",
                        "collection": "docs",
                        "chunk_index": 0,
                        "title": "Azure Guide",
                        "summary": "Explains Azure endpoints.",
                        "tags": ["azure", "endpoints"],
                        "page": 1,
                    },
                },
                {
                    "id": "vec-c",
                    "vector": [1.0],
                    "text": "Notebook chunk",
                    "metadata": {
                        "source_path": "notes/research.md",
                        "collection": "notes",
                        "chunk_index": 0,
                        "title": "Research Notes",
                        "summary": "Notes summary.",
                        "tags": ["research"],
                    },
                },
            ]
        ),
        encoding="utf-8",
    )


def test_get_document_summary_aggregates_metadata_by_file_stem(tmp_path: Path) -> None:
    write_records(tmp_path)
    history = tmp_path / "db" / "ingestion_history.db"
    with sqlite3.connect(history) as connection:
        connection.execute(
            "CREATE TABLE ingestion_history (file_hash TEXT, file_path TEXT, processed_at TEXT)"
        )
        connection.execute(
            "INSERT INTO ingestion_history (file_hash, file_path, processed_at) VALUES (?, ?, ?)",
            ("hash-azure", "docs/azure.pdf", "2026-05-08 10:00:00"),
        )

    result = get_document_summary({"doc_id": "azure", "data_dir": str(tmp_path)})

    document = result["structuredContent"]["document"]
    assert result["structuredContent"]["found"] is True
    assert document["title"] == "Azure Guide"
    assert document["summary"] == "Explains Azure endpoints. Covers deployments and versions."
    assert document["tags"] == ["azure", "endpoints", "deployments"]
    assert document["source"] == "docs/azure.pdf"
    assert document["collection"] == "docs"
    assert document["created_at"] == "2026-05-08 10:00:00"
    assert document["chunk_count"] == 2
    assert document["chunk_ids"] == ["vec-a", "vec-b"]
    assert document["pages"] == [1, 2]
    assert "## Azure Guide" in result["content"][0]["text"]


def test_get_document_summary_respects_collection_filter(tmp_path: Path) -> None:
    write_records(tmp_path)

    result = get_document_summary({"doc_id": "azure", "collection": "notes", "data_dir": str(tmp_path)})

    assert result["structuredContent"]["found"] is False
    assert result["structuredContent"]["document"] is None


def test_get_document_summary_falls_back_to_chunk_text_summary(tmp_path: Path) -> None:
    chroma = tmp_path / "db" / "chroma"
    chroma.mkdir(parents=True)
    (chroma / "records.json").write_text(
        json.dumps(
            [
                {
                    "id": "chunk-1",
                    "vector": [1.0],
                    "text": "Alpha beta gamma",
                    "metadata": {"source_path": "docs/plain.md", "collection": "docs", "chunk_index": 0},
                }
            ]
        ),
        encoding="utf-8",
    )

    result = get_document_summary({"doc_id": "plain.md", "data_dir": str(tmp_path)})

    document = result["structuredContent"]["document"]
    assert document["title"] == "plain"
    assert document["summary"] == "Alpha beta gamma"
    assert document["tags"] == []


def test_get_document_summary_returns_not_found_for_missing_index(tmp_path: Path) -> None:
    result = get_document_summary({"doc_id": "missing", "data_dir": str(tmp_path)})

    assert result["structuredContent"] == {
        "doc_id": "missing",
        "found": False,
        "document": None,
        "data_dir": str(tmp_path),
    }
    assert "Document not found" in result["content"][0]["text"]


def test_get_document_summary_rejects_invalid_arguments() -> None:
    with pytest.raises(ProtocolError, match="doc_id"):
        get_document_summary({"doc_id": ""})
    with pytest.raises(ProtocolError, match="data_dir"):
        get_document_summary({"doc_id": "a", "data_dir": ""})
    with pytest.raises(ProtocolError, match="collection"):
        get_document_summary({"doc_id": "a", "collection": ""})


def test_get_document_summary_tool_spec_has_mcp_shape() -> None:
    spec = get_document_summary_tool_spec()

    assert spec.name == "get_document_summary"
    assert spec.to_mcp_tool()["inputSchema"]["required"] == ["doc_id"]
