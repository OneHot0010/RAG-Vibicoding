"""Tests for core ingestion/retrieval data contracts."""

from __future__ import annotations

import json

import pytest

from core.types import (
    Chunk,
    ChunkRecord,
    CoreTypeError,
    Document,
    ImageRef,
    RetrievalResult,
    extract_image_placeholders,
    image_placeholder,
    normalize_image_refs,
)


def image_ref() -> ImageRef:
    return ImageRef(
        id="doc123_1_0",
        path="data/images/default/doc123_1_0.png",
        page=1,
        text_offset=12,
        text_length=len("[IMAGE: doc123_1_0]"),
        position={"x": 10, "y": 20, "width": 100, "height": 80},
    )


def metadata() -> dict:
    return {
        "source_path": "tests/fixtures/sample_documents/minimal.md",
        "doc_type": "markdown",
        "images": [image_ref().to_dict()],
    }


def test_document_serializes_to_stable_json_fields() -> None:
    document = Document(id="doc-1", text="Hello [IMAGE: doc123_1_0]", metadata=metadata())

    serialized = document.to_dict()

    assert list(serialized) == ["id", "text", "metadata"]
    assert serialized["metadata"]["source_path"] == "tests/fixtures/sample_documents/minimal.md"
    assert serialized["metadata"]["images"][0]["id"] == "doc123_1_0"
    assert Document.from_dict(json.loads(json.dumps(serialized))) == document


def test_chunk_serializes_offsets_and_source_ref() -> None:
    chunk = Chunk(
        id="doc-1_0000_abcd1234",
        text="Hello",
        metadata={"source_path": "sample.md", "chunk_index": 0},
        start_offset=0,
        end_offset=5,
        source_ref="doc-1",
    )

    serialized = chunk.to_dict()

    assert serialized == {
        "id": "doc-1_0000_abcd1234",
        "text": "Hello",
        "metadata": {"source_path": "sample.md", "chunk_index": 0},
        "start_offset": 0,
        "end_offset": 5,
        "source_ref": "doc-1",
    }
    assert Chunk.from_dict(serialized) == chunk


def test_chunk_record_serializes_optional_vectors() -> None:
    record = ChunkRecord(
        id="chunk-1",
        text="content",
        metadata={"source_path": "sample.md"},
        dense_vector=[0.1, 0.2],
        sparse_vector={"rag": 1.5},
    )

    serialized = record.to_dict()

    assert serialized["dense_vector"] == [0.1, 0.2]
    assert serialized["sparse_vector"] == {"rag": 1.5}
    assert ChunkRecord.from_dict(serialized) == record


def test_retrieval_result_serializes_stable_shape() -> None:
    result = RetrievalResult(
        chunk_id="chunk-1",
        score=0.75,
        text="retrieved text",
        metadata={"source_path": "sample.md", "collection": "docs"},
    )

    serialized = result.to_dict()

    assert serialized == {
        "chunk_id": "chunk-1",
        "score": 0.75,
        "text": "retrieved text",
        "metadata": {"source_path": "sample.md", "collection": "docs"},
    }
    assert RetrievalResult.from_dict(serialized) == result


def test_metadata_requires_source_path() -> None:
    with pytest.raises(CoreTypeError, match="metadata.source_path"):
        Document(id="doc-1", text="content", metadata={})


def test_metadata_images_are_normalized_and_validated() -> None:
    images = normalize_image_refs(
        [
            {
                "id": "image-1",
                "path": "data/images/default/image-1.png",
                "page": 2,
                "text_offset": 4,
                "text_length": 16,
                "position": {"bbox": [1, 2, 3, 4]},
            }
        ]
    )

    assert images == [
        {
            "id": "image-1",
            "path": "data/images/default/image-1.png",
            "page": 2,
            "text_offset": 4,
            "text_length": 16,
            "position": {"bbox": [1, 2, 3, 4]},
        }
    ]


def test_invalid_image_metadata_has_readable_error() -> None:
    with pytest.raises(CoreTypeError, match="ImageRef.id"):
        Document(
            id="doc-1",
            text="content",
            metadata={"source_path": "sample.md", "images": [{"path": "image.png"}]},
        )


def test_image_placeholder_helpers_use_canonical_format() -> None:
    placeholder = image_placeholder("doc123_1_0")

    assert placeholder == "[IMAGE: doc123_1_0]"
    assert extract_image_placeholders(f"before {placeholder} after [IMAGE: another-2]") == [
        "doc123_1_0",
        "another-2",
    ]


def test_chunk_offsets_must_be_valid() -> None:
    with pytest.raises(CoreTypeError, match="end_offset"):
        Chunk(
            id="chunk-1",
            text="content",
            metadata={"source_path": "sample.md"},
            start_offset=10,
            end_offset=2,
        )


def test_chunk_record_vectors_must_be_numeric() -> None:
    with pytest.raises(CoreTypeError, match="dense_vector"):
        ChunkRecord(
            id="chunk-1",
            text="content",
            metadata={"source_path": "sample.md"},
            dense_vector=[1.0, "bad"],  # type: ignore[list-item]
        )
