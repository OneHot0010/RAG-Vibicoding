"""Dashboard ingestion service helpers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from core.settings import Settings, load_settings
from core.trace import TraceCollector, TraceContext
from ingestion.chunking import DocumentChunker
from ingestion.embedding import BatchProcessor, DenseEncoder, SparseEncoder
from ingestion.pipeline import IngestionPipeline, IngestionPipelineResult
from ingestion.storage import BM25Indexer, ImageStorage, VectorUpserter
from libs.embedding import BaseEmbedding
from libs.loader import PdfLoader, SQLiteIntegrityChecker
from observability.dashboard.services.data_service import DataService
from observability.logger import write_trace


ProgressCallback = Callable[[str, int, int], None]
PipelineFactory = Callable[[Settings, Path], IngestionPipeline]


@dataclass(frozen=True)
class ProgressEvent:
    """One ingestion progress update."""

    stage: str
    current: int
    total: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize progress event."""
        return {"stage": self.stage, "current": self.current, "total": self.total}


@dataclass(frozen=True)
class DashboardIngestionResult:
    """Dashboard-friendly ingestion result."""

    source_path: str
    collection: str
    skipped: bool
    file_hash: str
    chunk_count: int
    image_count: int
    trace_id: str
    progress: list[ProgressEvent]

    def to_dict(self) -> dict[str, Any]:
        """Serialize ingestion result."""
        return {
            "source_path": self.source_path,
            "collection": self.collection,
            "skipped": self.skipped,
            "file_hash": self.file_hash,
            "chunk_count": self.chunk_count,
            "image_count": self.image_count,
            "trace_id": self.trace_id,
            "progress": [event.to_dict() for event in self.progress],
        }


class DashboardHashEmbedding(BaseEmbedding):
    """Deterministic local embedding backend for dashboard ingestion."""

    dimension = 8

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        """Return stable vectors without external API calls."""
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([round(digest[index] / 255.0, 6) for index in range(self.dimension)])
        return vectors


class IngestionService:
    """Coordinate Dashboard upload ingestion and document deletion."""

    def __init__(
        self,
        *,
        settings_path: str | Path = "config/settings.yaml",
        data_dir: str | Path = "data",
        data_service: DataService | None = None,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self.settings_path = Path(settings_path)
        self.data_dir = Path(data_dir)
        self.upload_dir = self.data_dir / "uploads"
        self.data_service = data_service or DataService(data_dir=self.data_dir)
        self.pipeline_factory = pipeline_factory or build_dashboard_pipeline

    def list_documents(self, collection: str | None = None) -> list[dict[str, Any]]:
        """Return documents available for management."""
        return self.data_service.list_documents(collection=collection)

    def save_upload(self, uploaded_file: Any) -> Path:
        """Persist a Streamlit uploaded file and return its local path."""
        filename = _safe_upload_name(str(getattr(uploaded_file, "name", "")))
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        path = self.upload_dir / filename
        path.write_bytes(_uploaded_bytes(uploaded_file))
        return path

    def ingest_file(
        self,
        path: str | Path,
        *,
        collection: str = "default",
        force: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> DashboardIngestionResult:
        """Run ingestion for one uploaded file."""
        settings = load_settings(self.settings_path)
        pipeline = self.pipeline_factory(settings, self.data_dir)
        progress_events: list[ProgressEvent] = []
        trace = TraceContext(trace_type="ingestion")
        trace.record_stage(
            "ingestion.dashboard.start",
            {"source_path": str(path), "collection": collection, "force": force},
        )

        def on_progress(stage: str, current: int, total: int) -> None:
            event = ProgressEvent(stage=stage, current=current, total=total)
            progress_events.append(event)
            if progress_callback is not None:
                progress_callback(stage, current, total)

        result = pipeline.run(path, collection=collection, force=force, trace=trace, on_progress=on_progress)
        trace.record_stage(
            "ingestion.dashboard.completed",
            {
                "source_path": result.source_path,
                "collection": result.collection,
                "skipped": result.skipped,
                "chunk_count": len(result.chunks),
                "image_count": len(result.image_records),
            },
        )
        self._persist_trace(settings, trace)
        return _dashboard_result(result, progress_events)

    def ingest_upload(
        self,
        uploaded_file: Any,
        *,
        collection: str = "default",
        force: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> DashboardIngestionResult:
        """Save and ingest a Streamlit uploaded file."""
        return self.ingest_file(
            self.save_upload(uploaded_file),
            collection=collection,
            force=force,
            progress_callback=progress_callback,
        )

    def delete_document(self, source_path: str, collection: str | None = None) -> dict[str, Any]:
        """Delete one managed document."""
        return self.data_service.delete_document(source_path, collection=collection)

    def _persist_trace(self, settings: Settings, trace: TraceContext) -> None:
        if not settings.observability.enabled:
            trace.finish()
            return
        TraceCollector(sink=lambda payload: write_trace(payload, log_file=settings.observability.log_file)).collect(trace)


def build_dashboard_pipeline(settings: Settings, data_dir: Path) -> IngestionPipeline:
    """Build a local dashboard ingestion pipeline."""
    data_root = Path(data_dir)
    db_root = data_root / "db"
    settings = replace(settings, vector_store=replace(settings.vector_store, persist_path=str(db_root / "chroma")))
    dense_encoder = DenseEncoder(settings, embedding=DashboardHashEmbedding())
    return IngestionPipeline(
        settings=settings,
        integrity_checker=SQLiteIntegrityChecker(db_root / "ingestion_history.db"),
        loader=PdfLoader(images_root=data_root / "images" / "_extracted"),
        chunker=DocumentChunker(settings),
        batch_processor=BatchProcessor(settings, dense_encoder=dense_encoder, sparse_encoder=SparseEncoder()),
        bm25_indexer=BM25Indexer(index_dir=db_root / "bm25"),
        vector_upserter=VectorUpserter(settings),
        image_storage=ImageStorage(images_root=data_root / "images", db_path=db_root / "image_index.db"),
    )


def _dashboard_result(result: IngestionPipelineResult, progress_events: list[ProgressEvent]) -> DashboardIngestionResult:
    return DashboardIngestionResult(
        source_path=result.source_path,
        collection=result.collection,
        skipped=result.skipped,
        file_hash=result.file_hash,
        chunk_count=len(result.chunks),
        image_count=len(result.image_records),
        trace_id=result.trace.trace_id,
        progress=progress_events,
    )


def _safe_upload_name(filename: str) -> str:
    name = Path(filename).name
    if not name:
        raise ValueError("uploaded file must have a name")
    if Path(name).suffix.lower() != ".pdf":
        raise ValueError("only PDF uploads are supported")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not safe:
        raise ValueError("uploaded file name is invalid")
    return safe


def _uploaded_bytes(uploaded_file: Any) -> bytes:
    if hasattr(uploaded_file, "getvalue"):
        payload = uploaded_file.getvalue()
    elif hasattr(uploaded_file, "read"):
        payload = uploaded_file.read()
    else:
        raise ValueError("uploaded file does not expose bytes")
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("uploaded file is empty")
    return payload


__all__ = [
    "DashboardHashEmbedding",
    "DashboardIngestionResult",
    "IngestionService",
    "ProgressEvent",
    "build_dashboard_pipeline",
]
