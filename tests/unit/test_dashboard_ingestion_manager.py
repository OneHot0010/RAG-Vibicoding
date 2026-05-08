"""Tests for dashboard ingestion manager page and service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.trace import TraceContext
from ingestion.pipeline import IngestionPipelineResult
from observability.dashboard.pages.ingestion_manager import ingestion_manager_model, render
from observability.dashboard.services.ingestion_service import IngestionService


class FakeIngestionService:
    def __init__(self) -> None:
        self.documents = [
            {
                "source_path": "docs/a.pdf",
                "collection": "docs",
                "doc_hash": "hash-a",
                "chunk_count": 2,
                "image_count": 1,
                "processed_at": "2026-05-08 10:00:00",
            }
        ]

    def list_documents(self, collection: str | None = None) -> list[dict[str, Any]]:
        if collection is None:
            return list(self.documents)
        return [document for document in self.documents if document["collection"] == collection]


class FakeUpload:
    name = "Demo File.pdf"

    def getvalue(self) -> bytes:
        return b"%PDF-1.4 demo"


class FakePipeline:
    def run(
        self,
        path: str | Path,
        collection: str = "default",
        force: bool = False,
        trace: TraceContext | None = None,
        on_progress: Any | None = None,
    ) -> IngestionPipelineResult:
        trace = trace or TraceContext(trace_type="ingestion")
        if on_progress is not None:
            on_progress("pipeline.start", 1, 2)
            on_progress("pipeline.completed", 2, 2)
        return IngestionPipelineResult(
            file_hash="hash-upload",
            source_path=str(path),
            collection=collection,
            skipped=force,
            document=None,
            chunks=[object(), object()],
            dense_records=[],
            sparse_records=[],
            vector_records=[],
            image_records=[],
            trace=trace,
        )


def write_settings(path: Path, trace_log: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
llm:
  provider: fake
  model: fake-chat
embedding:
  provider: fake
  model: fake-embedding
vision_llm:
  provider: fake
  model: fake-vision
splitter:
  strategy: recursive
  chunk_size: 100
  chunk_overlap: 10
vector_store:
  backend: chroma
  persist_path: ./data/db/chroma
retrieval:
  sparse_backend: bm25
  fusion_algorithm: rrf
  top_k_dense: 4
  top_k_sparse: 5
  top_k_final: 3
rerank:
  backend: none
  model: fake-reranker
  top_m: 2
evaluation:
  backends:
    - custom
  golden_test_set: ./golden.json
observability:
  enabled: false
  log_file: {trace_log.as_posix()}
""".strip(),
        encoding="utf-8",
    )


def test_ingestion_manager_model_formats_document_rows() -> None:
    model = ingestion_manager_model(FakeIngestionService())

    assert model["collections"] == ["docs"]
    assert model["document_rows"] == [
        {
            "source_path": "docs/a.pdf",
            "collection": "docs",
            "chunks": 2,
            "images": 1,
            "processed_at": "2026-05-08 10:00:00",
        }
    ]


def test_ingestion_manager_render_without_streamlit_returns_model() -> None:
    rendered = render(ingestion_service=FakeIngestionService())

    assert rendered["documents"][0]["source_path"] == "docs/a.pdf"


def test_ingestion_service_saves_upload_and_reports_progress(tmp_path: Path) -> None:
    settings_path = tmp_path / "config" / "settings.yaml"
    write_settings(settings_path, tmp_path / "logs" / "traces.jsonl")
    service = IngestionService(
        settings_path=settings_path,
        data_dir=tmp_path / "data",
        pipeline_factory=lambda settings, data_dir: FakePipeline(),
    )
    progress: list[tuple[str, int, int]] = []

    result = service.ingest_upload(
        FakeUpload(),
        collection="docs",
        force=True,
        progress_callback=lambda stage, current, total: progress.append((stage, current, total)),
    )

    assert (tmp_path / "data" / "uploads" / "Demo_File.pdf").read_bytes() == b"%PDF-1.4 demo"
    assert result.to_dict()["collection"] == "docs"
    assert result.chunk_count == 2
    assert result.skipped is True
    assert progress == [("pipeline.start", 1, 2), ("pipeline.completed", 2, 2)]
