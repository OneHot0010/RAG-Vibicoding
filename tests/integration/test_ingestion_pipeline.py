"""Integration tests for the MVP ingestion pipeline."""

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
from core.trace.trace_context import TraceContext
from core.types import Chunk, Document, ImageRef, image_placeholder
from ingestion.chunking import DocumentChunker
from ingestion.embedding import BatchProcessor, DenseEncoder, SparseEncoder
from ingestion.pipeline import IngestionPipeline, IngestionPipelineError
from ingestion.storage import BM25Indexer, ImageStorage, VectorUpserter
from ingestion.transform import BaseTransform
from libs.embedding import BaseEmbedding
from libs.loader import BaseLoader, LoaderError, SQLiteIntegrityChecker
from libs.splitter import BaseSplitter


class FakeLoader(BaseLoader):
    def __init__(self, image_path: Path | None = None) -> None:
        self.image_path = image_path
        self.calls = 0

    def load(self, path: str | Path) -> Document:
        self.calls += 1
        text = "Alpha retrieval systems use vectors.\n\nBeta retrieval systems use bm25."
        images: list[dict[str, Any]] = []
        if self.image_path is not None:
            image_id = "img-0"
            placeholder = image_placeholder(image_id)
            text = f"{text}\n\n{placeholder}"
            images = [
                ImageRef(
                    id=image_id,
                    path=str(self.image_path),
                    page=1,
                    text_offset=text.index("[IMAGE:"),
                    text_length=len(placeholder),
                ).to_dict()
            ]
        return Document(
            id="doc-1",
            text=text,
            metadata={"source_path": str(path), "file_hash": "doc-hash", "images": images},
        )


class FailingLoader(BaseLoader):
    def load(self, path: str | Path) -> Document:
        raise LoaderError("cannot parse")


class SentenceSplitter(BaseSplitter):
    def split_text(self, text: str, trace: Any | None = None) -> list[str]:
        return [part.strip() for part in text.split("\n\n") if part.strip()]


class FakeEmbedding(BaseEmbedding):
    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        return [[float(index + 1), float(len(text) % 7 + 1)] for index, text in enumerate(texts)]


class MetadataTransform(BaseTransform):
    def transform(self, chunks: list[Chunk], trace: Any | None = None) -> list[Chunk]:
        transformed = []
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            metadata["transformed"] = True
            transformed.append(
                Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=metadata,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    source_ref=chunk.source_ref,
                )
            )
        return transformed


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
    dense_encoder = DenseEncoder(settings, embedding=FakeEmbedding())
    return IngestionPipeline(
        settings=settings,
        integrity_checker=SQLiteIntegrityChecker(tmp_path / "ingestion_history.db"),
        loader=loader,
        chunker=DocumentChunker(settings, splitter=SentenceSplitter()),
        transforms=[MetadataTransform()],
        batch_processor=BatchProcessor(
            settings,
            batch_size=2,
            dense_encoder=dense_encoder,
            sparse_encoder=SparseEncoder(),
        ),
        bm25_indexer=BM25Indexer(index_dir=tmp_path / "bm25"),
        vector_upserter=VectorUpserter(settings),
        image_storage=ImageStorage(images_root=tmp_path / "images", db_path=tmp_path / "image_index.db"),
    )


def test_ingestion_pipeline_writes_all_mvp_outputs(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF test")
    source_image = tmp_path / "raw" / "img.png"
    source_image.parent.mkdir()
    source_image.write_bytes(b"png")
    trace = TraceContext()

    result = make_pipeline(tmp_path, FakeLoader(image_path=source_image)).run(source, collection="docs", trace=trace)

    assert result.skipped is False
    assert result.document is not None
    assert len(result.chunks) == 3
    assert len(result.dense_records) == 3
    assert len(result.sparse_records) == 3
    assert len(result.vector_records) == 3
    assert len(result.image_records) == 1
    assert (tmp_path / "chroma" / "records.json").is_file()
    assert (tmp_path / "bm25" / "index.json").is_file()
    assert Path(result.image_records[0].file_path).read_bytes() == b"png"
    assert result.image_records[0].collection == "docs"
    assert result.chunks[0].metadata["collection"] == "docs"
    assert result.chunks[0].metadata["transformed"] is True
    assert result.to_dict()["chunk_count"] == 3
    assert "pipeline.completed" in [stage["name"] for stage in trace.stages]


def test_pipeline_skips_successful_file_without_force(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF test")
    loader = FakeLoader()
    pipeline = make_pipeline(tmp_path, loader)

    first = pipeline.run(source, collection="docs")
    second = pipeline.run(source, collection="docs")

    assert first.skipped is False
    assert second.skipped is True
    assert loader.calls == 1


def test_pipeline_force_reruns_successful_file(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF test")
    loader = FakeLoader()
    pipeline = make_pipeline(tmp_path, loader)

    pipeline.run(source, collection="docs")
    rerun = pipeline.run(source, collection="docs", force=True)

    assert rerun.skipped is False
    assert loader.calls == 2


def test_pipeline_failure_marks_integrity_record_and_names_stage(tmp_path: Path) -> None:
    source = tmp_path / "broken.pdf"
    source.write_bytes(b"%PDF broken")
    pipeline = make_pipeline(tmp_path, FailingLoader())

    with pytest.raises(IngestionPipelineError, match="loader.load failed"):
        pipeline.run(source, collection="docs")

    file_hash = pipeline.integrity_checker.compute_sha256(source)
    record = pipeline.integrity_checker.get_record(file_hash)
    assert record["status"] == "failed"
    assert "loader.load failed" in record["error_msg"]
