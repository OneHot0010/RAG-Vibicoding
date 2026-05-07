"""Tests for Document -> Chunk business adaptation."""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import (
    EmbeddingSettings,
    EvaluationSettings,
    LLMSettings,
    ObservabilitySettings,
    RerankSettings,
    RetrievalSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
    VisionLLMSettings,
)
from core.types import Chunk, Document, ImageRef, image_placeholder
from ingestion.chunking import DocumentChunker, DocumentChunkerError
from libs.splitter import BaseSplitter, SplitterFactory


class FakeSplitter(BaseSplitter):
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, Any | None]] = []

    def split_text(self, text: str, trace: Any | None = None) -> list[str]:
        self.calls.append((text, trace))
        return self.chunks


class LengthSplitter(BaseSplitter):
    def __init__(self, settings: SplitterSettings) -> None:
        self.settings = settings

    def split_text(self, text: str, trace: Any | None = None) -> list[str]:
        return [
            text[index : index + self.settings.chunk_size]
            for index in range(0, len(text), self.settings.chunk_size)
            if text[index : index + self.settings.chunk_size]
        ]


def make_settings(chunk_size: int = 100) -> Settings:
    return Settings(
        llm=LLMSettings(provider="fake", model="fake-chat"),
        embedding=EmbeddingSettings(provider="fake", model="fake-embedding"),
        vision_llm=VisionLLMSettings(provider="fake", model="fake-vision"),
        splitter=SplitterSettings(strategy="length", chunk_size=chunk_size, chunk_overlap=0),
        vector_store=VectorStoreSettings(backend="fake", persist_path="./tmp/vector"),
        retrieval=RetrievalSettings(
            sparse_backend="bm25",
            fusion_algorithm="rrf",
            top_k_dense=20,
            top_k_sparse=20,
            top_k_final=10,
        ),
        rerank=RerankSettings(backend="none"),
        evaluation=EvaluationSettings(backends=["custom"], golden_test_set="./golden.json"),
        observability=ObservabilitySettings(enabled=True, log_file="./logs/traces.jsonl"),
    )


def make_document() -> Document:
    first_image = ImageRef(
        id="img-1",
        path="data/images/doc/img-1.png",
        page=1,
        text_offset=6,
        text_length=len(image_placeholder("img-1")),
    )
    second_image = ImageRef(
        id="img-2",
        path="data/images/doc/img-2.png",
        page=2,
        text_offset=40,
        text_length=len(image_placeholder("img-2")),
    )
    text = f"Intro {image_placeholder('img-1')}\n\nDetails {image_placeholder('img-2')}\n\nPlain tail"
    return Document(
        id="doc-abc",
        text=text,
        metadata={
            "source_path": "docs/sample.pdf",
            "doc_type": "pdf",
            "title": "Sample",
            "images": [first_image.to_dict(), second_image.to_dict()],
        },
    )


def test_split_document_converts_splitter_output_to_chunks() -> None:
    document = make_document()
    splitter = FakeSplitter(["Intro " + image_placeholder("img-1"), "Plain tail"])

    chunks = DocumentChunker(make_settings(), splitter=splitter).split_document(document)

    assert len(chunks) == 2
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert [chunk.rank if hasattr(chunk, "rank") else chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1]
    assert [chunk.source_ref for chunk in chunks] == ["doc-abc", "doc-abc"]
    assert splitter.calls == [(document.text, None)]


def test_chunk_ids_are_unique_and_deterministic() -> None:
    document = make_document()
    splitter = FakeSplitter(["alpha", "beta", "alpha"])
    chunker = DocumentChunker(make_settings(), splitter=splitter)

    first = chunker.split_document(document)
    second = chunker.split_document(document)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert len({chunk.id for chunk in first}) == 3
    assert first[0].id.startswith("doc-abc_0000_")
    assert first[2].id.startswith("doc-abc_0002_")


def test_offsets_are_located_in_source_document_order() -> None:
    document = Document(id="doc", text="repeat\nmiddle\nrepeat", metadata={"source_path": "docs/a.md"})
    splitter = FakeSplitter(["repeat", "repeat"])

    chunks = DocumentChunker(make_settings(), splitter=splitter).split_document(document)

    assert [(chunk.start_offset, chunk.end_offset) for chunk in chunks] == [(0, 6), (14, 20)]


def test_metadata_is_inherited_with_chunk_index() -> None:
    document = Document(
        id="doc",
        text="alpha",
        metadata={"source_path": "docs/a.md", "doc_type": "markdown", "title": "A"},
    )

    chunk = DocumentChunker(make_settings(), splitter=FakeSplitter(["alpha"])).split_document(document)[0]

    assert chunk.metadata == {
        "source_path": "docs/a.md",
        "doc_type": "markdown",
        "title": "A",
        "chunk_index": 0,
    }


def test_images_are_distributed_only_to_chunks_that_reference_them() -> None:
    document = make_document()
    splitter = FakeSplitter(
        [
            "Intro " + image_placeholder("img-1"),
            "Details " + image_placeholder("img-2"),
            "Plain tail",
        ]
    )

    chunks = DocumentChunker(make_settings(), splitter=splitter).split_document(document)

    assert chunks[0].metadata["image_refs"] == ["img-1"]
    assert [image["id"] for image in chunks[0].metadata["images"]] == ["img-1"]
    assert chunks[1].metadata["image_refs"] == ["img-2"]
    assert [image["id"] for image in chunks[1].metadata["images"]] == ["img-2"]
    assert "images" not in chunks[2].metadata
    assert "image_refs" not in chunks[2].metadata


def test_splitter_is_created_from_settings() -> None:
    SplitterFactory.clear()
    SplitterFactory.register("length", LengthSplitter)
    document = Document(id="doc", text="abcdefghij", metadata={"source_path": "docs/a.md"})

    small = DocumentChunker(make_settings(chunk_size=3)).split_document(document)
    large = DocumentChunker(make_settings(chunk_size=6)).split_document(document)

    assert [chunk.text for chunk in small] == ["abc", "def", "ghi", "j"]
    assert [chunk.text for chunk in large] == ["abcdef", "ghij"]


def test_chunks_are_serializable_core_type_contracts() -> None:
    document = Document(id="doc", text="alpha", metadata={"source_path": "docs/a.md"})

    chunk = DocumentChunker(make_settings(), splitter=FakeSplitter(["alpha"])).split_document(document)[0]

    assert Chunk.from_dict(chunk.to_dict()) == chunk


def test_non_document_input_has_readable_error() -> None:
    with pytest.raises(DocumentChunkerError, match="Document"):
        DocumentChunker(make_settings(), splitter=FakeSplitter([])).split_document("not-document")  # type: ignore[arg-type]
