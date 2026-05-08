"""Ingestion pipeline package."""

from ingestion.document_manager import CollectionStats, DeleteResult, DocumentDetail, DocumentInfo, DocumentManager
from ingestion.pipeline import IngestionPipeline, IngestionPipelineError, IngestionPipelineResult

__all__ = [
    "CollectionStats",
    "DeleteResult",
    "DocumentDetail",
    "DocumentInfo",
    "DocumentManager",
    "IngestionPipeline",
    "IngestionPipelineError",
    "IngestionPipelineResult",
]
