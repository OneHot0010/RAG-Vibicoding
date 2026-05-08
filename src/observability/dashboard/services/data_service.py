"""Dashboard data service wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ingestion.document_manager import DeleteResult, DocumentDetail, DocumentInfo, DocumentManager


class DataService:
    """Expose document manager operations in UI-friendly dictionary shapes."""

    def __init__(self, data_dir: str | Path = "data", document_manager: DocumentManager | None = None) -> None:
        self.document_manager = document_manager or DocumentManager(data_dir=data_dir)

    def list_documents(self, collection: str | None = None) -> list[dict[str, Any]]:
        """Return document summaries."""
        return [document.to_dict() for document in self.document_manager.list_documents(collection=collection)]

    def get_document_detail(self, doc_id: str) -> dict[str, Any]:
        """Return one document detail."""
        return self.document_manager.get_document_detail(doc_id).to_dict()

    def delete_document(self, source_path: str, collection: str | None = None) -> dict[str, Any]:
        """Delete a document and return deletion counts."""
        return self.document_manager.delete_document(source_path, collection=collection).to_dict()

    def get_collection_stats(self, collection: str | None = None) -> dict[str, Any]:
        """Return aggregate collection stats."""
        return self.document_manager.get_collection_stats(collection=collection).to_dict()


__all__ = ["DataService", "DeleteResult", "DocumentDetail", "DocumentInfo", "DocumentManager"]
