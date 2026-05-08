"""End-to-end ingestion pipeline orchestration."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord, Document, normalize_image_refs
from ingestion.chunking import DocumentChunker
from ingestion.embedding import BatchProcessingResult, BatchProcessor, SparseEncodingResult
from ingestion.storage import BM25Indexer, ImageRecord, ImageStorage, VectorUpserter
from ingestion.transform import ChunkRefiner, ImageCaptioner, MetadataEnricher
from ingestion.transform.base_transform import BaseTransform
from libs.loader import BaseLoader, PdfLoader, SQLiteIntegrityChecker
from libs.vector_store import VectorRecord


class IngestionPipelineError(RuntimeError):
    """Raised when a pipeline stage fails."""


@dataclass(frozen=True)
class IngestionPipelineResult:
    """Structured result returned by a pipeline run."""

    file_hash: str
    source_path: str
    collection: str
    skipped: bool
    document: Document | None
    chunks: list[Chunk]
    dense_records: list[ChunkRecord]
    sparse_records: list[ChunkRecord]
    vector_records: list[VectorRecord]
    image_records: list[ImageRecord]
    trace: TraceContext

    def to_dict(self) -> dict[str, Any]:
        """Serialize the pipeline result for CLI or trace output."""
        return {
            "file_hash": self.file_hash,
            "source_path": self.source_path,
            "collection": self.collection,
            "skipped": self.skipped,
            "document_id": self.document.id if self.document is not None else None,
            "chunk_count": len(self.chunks),
            "dense_record_count": len(self.dense_records),
            "sparse_record_count": len(self.sparse_records),
            "vector_record_count": len(self.vector_records),
            "image_record_count": len(self.image_records),
            "trace_id": self.trace.trace_id,
        }


class IngestionPipeline:
    """Serially execute integrity, loading, transforms, encoding, and storage."""

    def __init__(
        self,
        settings: Settings,
        integrity_checker: SQLiteIntegrityChecker | None = None,
        loader: BaseLoader | None = None,
        chunker: DocumentChunker | None = None,
        transforms: list[BaseTransform] | None = None,
        batch_processor: BatchProcessor | None = None,
        bm25_indexer: BM25Indexer | None = None,
        vector_upserter: VectorUpserter | None = None,
        image_storage: ImageStorage | None = None,
    ) -> None:
        self.settings = settings
        self.integrity_checker = integrity_checker or SQLiteIntegrityChecker()
        self.loader = loader or PdfLoader()
        self.chunker = chunker or DocumentChunker(settings)
        self.transforms = transforms if transforms is not None else [
            ChunkRefiner(settings),
            MetadataEnricher(settings),
            ImageCaptioner(settings),
        ]
        self.batch_processor = batch_processor or BatchProcessor(settings)
        self.bm25_indexer = bm25_indexer or BM25Indexer()
        self.vector_upserter = vector_upserter or VectorUpserter(settings)
        self.image_storage = image_storage or ImageStorage()

    def run(
        self,
        path: str | Path,
        collection: str = "default",
        force: bool = False,
        trace: TraceContext | None = None,
    ) -> IngestionPipelineResult:
        """Run ingestion for one source file."""
        trace = trace or TraceContext(trace_type="ingestion")
        source_path = Path(path)
        _record_trace(trace, "pipeline.start", {"source_path": str(source_path), "collection": collection, "force": force})

        file_hash = self._stage("integrity.compute_sha256", lambda: self.integrity_checker.compute_sha256(source_path), trace)
        if not force and self._stage("integrity.should_skip", lambda: self.integrity_checker.should_skip(file_hash), trace):
            _record_trace(trace, "pipeline.skipped", {"file_hash": file_hash})
            return IngestionPipelineResult(
                file_hash=file_hash,
                source_path=str(source_path),
                collection=collection,
                skipped=True,
                document=None,
                chunks=[],
                dense_records=[],
                sparse_records=[],
                vector_records=[],
                image_records=[],
                trace=trace,
            )

        try:
            document = self._stage("loader.load", lambda: self.loader.load(source_path), trace)
            document = _with_collection(document, collection)
            chunks = self._stage("chunker.split_document", lambda: self.chunker.split_document(document, trace=trace), trace)
            chunks = self._stage("transforms.apply", lambda: self._apply_transforms(chunks, trace), trace)
            batch_result = self._stage("batch_processor.process", lambda: self.batch_processor.process(chunks, trace=trace), trace)
            sparse_result = _combine_sparse_results(batch_result)
            self._stage("bm25_indexer.build", lambda: self.bm25_indexer.build(sparse_result, trace=trace), trace)
            self._stage("bm25_indexer.save", self.bm25_indexer.save, trace)
            vector_records = self._stage(
                "vector_upserter.upsert",
                lambda: self.vector_upserter.upsert(batch_result.dense_records, trace=trace),
                trace,
            )
            image_records = self._stage(
                "image_storage.index",
                lambda: self._store_images(document, collection, file_hash, trace),
                trace,
            )
            self._stage(
                "integrity.mark_success",
                lambda: self.integrity_checker.mark_success(
                    file_hash,
                    source_path,
                    file_size=source_path.stat().st_size,
                    chunk_count=len(chunks),
                ),
                trace,
            )
            _record_trace(
                trace,
                "pipeline.completed",
                {"file_hash": file_hash, "chunk_count": len(chunks), "image_count": len(image_records)},
            )
            return IngestionPipelineResult(
                file_hash=file_hash,
                source_path=str(source_path),
                collection=collection,
                skipped=False,
                document=document,
                chunks=chunks,
                dense_records=batch_result.dense_records,
                sparse_records=batch_result.sparse_records,
                vector_records=vector_records,
                image_records=image_records,
                trace=trace,
            )
        except Exception as exc:
            try:
                self.integrity_checker.mark_failed(file_hash, str(exc), file_path=source_path)
            except Exception:
                pass
            if isinstance(exc, IngestionPipelineError):
                raise
            raise IngestionPipelineError(f"pipeline failed: {exc}") from exc

    def _stage(self, name: str, action: Any, trace: TraceContext) -> Any:
        try:
            result = action()
        except Exception as exc:
            _record_trace(trace, "pipeline.failed", {"stage": name, "error": str(exc)})
            raise IngestionPipelineError(f"{name} failed: {exc}") from exc
        _record_trace(trace, name, _stage_data(result))
        return result

    def _apply_transforms(self, chunks: list[Chunk], trace: TraceContext) -> list[Chunk]:
        transformed = chunks
        for transform in self.transforms:
            transformed = transform.transform(transformed, trace=trace)
        return transformed

    def _store_images(
        self,
        document: Document,
        collection: str,
        file_hash: str,
        trace: TraceContext,
    ) -> list[ImageRecord]:
        image_records: list[ImageRecord] = []
        for image in normalize_image_refs(document.metadata.get("images", [])):
            source = Path(image["path"])
            if not source.is_file():
                raise IngestionPipelineError(f"image file not found: {source}")
            image_records.append(
                self.image_storage.save_file(
                    image_id=image["id"],
                    source_path=source,
                    collection=collection,
                    doc_hash=file_hash,
                    page_num=image.get("page"),
                    trace=trace,
                )
            )
        return image_records


def _with_collection(document: Document, collection: str) -> Document:
    metadata = copy.deepcopy(document.metadata)
    metadata["collection"] = collection
    return Document(id=document.id, text=document.text, metadata=metadata)


def _combine_sparse_results(batch_result: BatchProcessingResult) -> SparseEncodingResult:
    records = batch_result.sparse_records
    document_frequency: dict[str, int] = {}
    total_doc_length = 0
    for record in records:
        sparse_vector = record.sparse_vector or {}
        for term in sparse_vector:
            document_frequency[term] = document_frequency.get(term, 0) + 1
        total_doc_length += int(record.metadata.get("doc_length") or sum(sparse_vector.values()))
    average_doc_length = total_doc_length / len(records) if records else 0.0
    return SparseEncodingResult(
        records=records,
        document_frequency=dict(sorted(document_frequency.items())),
        total_chunks=len(records),
        average_doc_length=average_doc_length,
    )


def _stage_data(result: Any) -> dict[str, Any]:
    if isinstance(result, bool):
        return {"result": result}
    if isinstance(result, str):
        return {"value": result}
    if isinstance(result, Document):
        return {"document_id": result.id, "text_length": len(result.text)}
    if isinstance(result, list):
        return {"count": len(result)}
    if isinstance(result, BatchProcessingResult):
        return {"batch_count": len(result.batches), "chunk_count": len(result.dense_records)}
    if isinstance(result, Path):
        return {"path": str(result)}
    return {}


def _record_trace(trace: Any | None, name: str, data: dict[str, Any]) -> None:
    if hasattr(trace, "record_stage"):
        trace.record_stage(name, data)
