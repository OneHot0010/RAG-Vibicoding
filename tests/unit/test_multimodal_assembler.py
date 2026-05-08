"""Tests for MCP multimodal response assembly."""

from __future__ import annotations

import base64
import sqlite3
from pathlib import Path

from core.response import MultimodalAssembler, ResponseBuilder
from core.types import RetrievalResult


def hit(metadata: dict, chunk_id: str = "chunk-a") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=0.9,
        text="Chunk with image.",
        metadata={"source_path": "docs/a.pdf", **metadata},
    )


def test_assembler_reads_direct_chunk_image_path(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"png-bytes")

    assembly = MultimodalAssembler(data_dir=tmp_path).assemble(
        [
            hit(
                {
                    "image_refs": ["img-1"],
                    "images": [{"id": "img-1", "path": str(image_path)}],
                }
            )
        ]
    )

    assert assembly.skipped == []
    assert len(assembly.images) == 1
    content = assembly.to_content()[0]
    assert content == {
        "type": "image",
        "data": base64.b64encode(b"png-bytes").decode("ascii"),
        "mimeType": "image/png",
    }
    assert assembly.to_dict()["images"][0]["image_id"] == "img-1"


def test_assembler_uses_image_index_when_chunk_only_has_refs(tmp_path: Path) -> None:
    image_path = tmp_path / "stored.jpg"
    image_path.write_bytes(b"jpg-bytes")
    db_path = tmp_path / "db" / "image_index.db"
    db_path.parent.mkdir()
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE image_index (image_id TEXT PRIMARY KEY, file_path TEXT)")
        connection.execute(
            "INSERT INTO image_index (image_id, file_path) VALUES (?, ?)",
            ("img-indexed", str(image_path)),
        )

    assembly = MultimodalAssembler(data_dir=tmp_path).assemble([hit({"image_refs": ["img-indexed"]})])

    assert assembly.images[0].mime_type == "image/jpeg"
    assert assembly.images[0].path == str(image_path)
    assert assembly.to_content()[0]["data"] == base64.b64encode(b"jpg-bytes").decode("ascii")


def test_assembler_reports_missing_images_without_failing(tmp_path: Path) -> None:
    assembly = MultimodalAssembler(data_dir=tmp_path).assemble([hit({"image_refs": ["missing"]})])

    assert assembly.images == []
    assert assembly.skipped == [{"image_id": "missing", "chunk_id": "chunk-a", "reason": "not_found"}]


def test_response_builder_appends_image_content_and_structured_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"jpg-bytes")

    response = ResponseBuilder(multimodal_assembler=MultimodalAssembler(data_dir=tmp_path)).build(
        [
            hit(
                {
                    "image_refs": ["img-1"],
                    "images": [{"id": "img-1", "path": str(image_path)}],
                }
            )
        ],
        "show image",
    )
    payload = response.to_dict()

    assert [item["type"] for item in payload["content"]] == ["text", "image"]
    assert payload["content"][1]["mimeType"] == "image/jpeg"
    assert payload["structuredContent"]["multimodal"]["images"][0]["image_id"] == "img-1"
