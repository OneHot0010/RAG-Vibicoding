"""Tests for IngestionPipeline progress callbacks."""

from __future__ import annotations

from pathlib import Path
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
from core.types import Chunk, Document
from ingestion.chunking import DocumentChunker
from ingestion.embedding import BatchProcessor, DenseEncoder, SparseEncoder
from ingestion.pipeline import IngestionPipeline, IngestionPipelineError
from ingestion.storage import BM25Indexer, ImageStorage, VectorUpserter
from ingestion.transform import BaseTransform
from libs.embedding import BaseEmbedding
from libs.loader import BaseLoader, LoaderError, SQLiteIntegrityChecker
from libs.splitter import BaseSplitter


class FakeLoader(BaseLoader):
    def __init__(self) -> None:
        self.calls = 0

    def load(self, path: str | Path) -> Document:
        self.calls += 1
        return Document(
            id="doc-1",
            text="Alpha retrieval systems.\n\nBeta vector stores.",
            metadata={"source_path": str(path), "file_hash": "doc-hash", "images": []},
        )


class FailingLoader(BaseLoader):
    def load(self, path: str | Path) -> Document:
        raise LoaderError("cannot parse")


class SentenceSplitter(BaseSplitter):
    def split_text(self, text: str, trace: Any | None = None) -> list[str]:
        return [part.strip() for part in text.split("\n\n") if part.strip()]


class FakeEmbedding(BaseEmbedding):
    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        return [[float(index + 1), 1.0] for index, _ in enumerate(texts)]


class PassthroughTransform(BaseTransform):
    def transform(self, chunks: list[Chunk], trace: Any | None = None) -> list[Chunk]:
        return chunks


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        llm=LLMSettings(provider="fake", model="fake-chat"),
        embedding=EmbeddingSettings(provider="fake", model="fake-embedding"),
        vision_llm=VisionLLMSettings(provider="fake", model="fake-vision"),
        splitter=SplitterSettings(strategy="fake", chunk_size=100, chunk_overlap=0),
        vector_store=VectorStoreSettings(backend="chroma", persist_path=str(tmp_path / "chroma")),
        retrieval=RetrievalSettings("bm25", "rrf", 20, 20, 10),
        rerank=RerankSettings(backend="none"),
        evaluation=EvaluationSettings(backends=["custom"], golden_test_set="./golden.json"),
        observability=ObservabilitySettings(enabled=True, log_file="./logs/traces.jsonl"),
        raw={"ingestion": {"batch_size": 2}},
    )


def make_pipeline(tmp_path: Path, loader: BaseLoader) -> IngestionPipeline:
    settings = make_settings(tmp_path)
    return IngestionPipeline(
        settings=settings,
        integrity_checker=SQLiteIntegrityChecker(tmp_path / "ingestion_history.db"),
        loader=loader,
        chunker=DocumentChunker(settings, splitter=SentenceSplitter()),
        transforms=[PassthroughTransform()],
        batch_processor=BatchProcessor(
            settings,
            batch_size=2,
            dense_encoder=DenseEncoder(settings, embedding=FakeEmbedding()),
            sparse_encoder=SparseEncoder(),
        ),
        bm25_indexer=BM25Indexer(index_dir=tmp_path / "bm25"),
        vector_upserter=VectorUpserter(settings),
        image_storage=ImageStorage(images_root=tmp_path / "images", db_path=tmp_path / "image_index.db"),
    )


def test_pipeline_progress_reports_successful_stage_sequence(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF test")
    events: list[tuple[str, int, int]] = []

    result = make_pipeline(tmp_path, FakeLoader()).run(
        source,
        collection="docs",
        on_progress=lambda stage, current, total: events.append((stage, current, total)),
    )

    assert result.skipped is False
    assert [event[0] for event in events] == [
        "pipeline.start",
        "integrity.compute_sha256",
        "integrity.should_skip",
        "loader.load",
        "chunker.split_document",
        "transforms.apply",
        "batch_processor.process",
        "bm25_indexer.build",
        "bm25_indexer.save",
        "vector_upserter.upsert",
        "image_storage.index",
        "integrity.mark_success",
        "pipeline.completed",
    ]
    assert events[-1] == ("pipeline.completed", 13, 13)
    assert [event[1] for event in events] == list(range(1, 14))
    assert all(event[2] == 13 for event in events)


def test_pipeline_progress_completes_when_file_is_skipped(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF test")
    loader = FakeLoader()
    pipeline = make_pipeline(tmp_path, loader)
    pipeline.run(source, collection="docs")
    events: list[tuple[str, int, int]] = []

    result = pipeline.run(
        source,
        collection="docs",
        on_progress=lambda stage, current, total: events.append((stage, current, total)),
    )

    assert result.skipped is True
    assert [event[0] for event in events] == [
        "pipeline.start",
        "integrity.compute_sha256",
        "integrity.should_skip",
        "pipeline.skipped",
    ]
    assert events[-1] == ("pipeline.skipped", 13, 13)


def test_pipeline_progress_completes_on_failure(tmp_path: Path) -> None:
    source = tmp_path / "broken.pdf"
    source.write_bytes(b"%PDF broken")
    events: list[tuple[str, int, int]] = []

    with pytest.raises(IngestionPipelineError, match="loader.load failed"):
        make_pipeline(tmp_path, FailingLoader()).run(
            source,
            collection="docs",
            on_progress=lambda stage, current, total: events.append((stage, current, total)),
        )

    assert [event[0] for event in events] == [
        "pipeline.start",
        "integrity.compute_sha256",
        "integrity.should_skip",
        "pipeline.failed",
    ]
    assert events[-1] == ("pipeline.failed", 13, 13)


def test_pipeline_progress_is_optional(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF test")

    result = make_pipeline(tmp_path, FakeLoader()).run(source, collection="docs")

    assert result.skipped is False
