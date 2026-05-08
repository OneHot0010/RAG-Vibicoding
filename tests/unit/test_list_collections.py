"""Tests for the list_collections MCP tool."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mcp_server.protocol_handler import ProtocolError
from mcp_server.tools.list_collections import list_collections, list_collections_tool_spec


def test_list_collections_returns_empty_message_for_missing_data_dir(tmp_path: Path) -> None:
    result = list_collections({"data_dir": str(tmp_path / "missing")})

    assert result["structuredContent"] == {
        "collections": [],
        "total_collections": 0,
        "data_dir": str(tmp_path / "missing"),
    }
    assert "未找到知识库集合" in result["content"][0]["text"]


def test_list_collections_scans_documents_directory(tmp_path: Path) -> None:
    docs = tmp_path / "documents" / "docs"
    notes = tmp_path / "documents" / "notes"
    docs.mkdir(parents=True)
    notes.mkdir(parents=True)
    (docs / "a.pdf").write_bytes(b"pdf")
    (docs / ".gitkeep").write_text("", encoding="utf-8")
    (notes / "b.md").write_text("note", encoding="utf-8")

    result = list_collections({"data_dir": str(tmp_path)})

    collections = result["structuredContent"]["collections"]
    assert [item["name"] for item in collections] == ["docs", "notes"]
    assert collections[0]["document_count"] == 1
    assert collections[0]["chunk_count"] == 0
    assert collections[0]["paths"] == [str(docs)]
    assert "Knowledge Hub Collections" in result["content"][0]["text"]


def test_list_collections_merges_vector_and_image_stats(tmp_path: Path) -> None:
    chroma = tmp_path / "db" / "chroma"
    chroma.mkdir(parents=True)
    (chroma / "records.json").write_text(
        json.dumps(
            [
                {"id": "a", "vector": [1.0], "text": "alpha", "metadata": {"collection": "docs", "source_path": "a.pdf"}},
                {"id": "b", "vector": [1.0], "text": "beta", "metadata": {"collection": "docs", "source_path": "a.pdf"}},
                {"id": "c", "vector": [1.0], "text": "gamma", "metadata": {"collection": "notes", "source": "note.md"}},
            ]
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "db" / "image_index.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE image_index (image_id TEXT, collection TEXT)")
        connection.executemany(
            "INSERT INTO image_index (image_id, collection) VALUES (?, ?)",
            [("img-a", "docs"), ("img-b", "docs"), ("img-c", "notes")],
        )

    result = list_collections({"data_dir": str(tmp_path)})

    collections = result["structuredContent"]["collections"]
    assert collections == [
        {
            "name": "docs",
            "description": "Indexed collection",
            "document_count": 1,
            "chunk_count": 2,
            "image_count": 2,
            "paths": [],
            "sources": ["a.pdf"],
        },
        {
            "name": "notes",
            "description": "Indexed collection",
            "document_count": 1,
            "chunk_count": 1,
            "image_count": 1,
            "paths": [],
            "sources": ["note.md"],
        },
    ]


def test_list_collections_rejects_invalid_data_dir() -> None:
    with pytest.raises(ProtocolError, match="data_dir"):
        list_collections({"data_dir": ""})


def test_list_collections_tool_spec_has_mcp_shape() -> None:
    spec = list_collections_tool_spec()

    assert spec.name == "list_collections"
    assert spec.to_mcp_tool()["inputSchema"] == {"type": "object", "properties": {}}
