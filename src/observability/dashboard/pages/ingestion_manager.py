"""Ingestion management page for uploads, progress, and deletion."""

from __future__ import annotations

from typing import Any

from observability.dashboard.services.ingestion_service import IngestionService


def ingestion_manager_model(
    ingestion_service: IngestionService | None = None,
    *,
    collection: str | None = None,
) -> dict[str, Any]:
    """Return display data for the ingestion manager page."""
    service = ingestion_service or IngestionService()
    documents = service.list_documents(collection=collection)
    collections = sorted({str(document["collection"]) for document in service.list_documents() if document.get("collection")})
    return {
        "collections": collections,
        "selected_collection": collection,
        "documents": documents,
        "document_rows": [_document_row(document) for document in documents],
    }


def render(
    st: Any | None = None,
    *,
    ingestion_service: IngestionService | None = None,
) -> dict[str, Any]:
    """Render the ingestion manager page and return its model for tests."""
    service = ingestion_service or IngestionService()
    model = ingestion_manager_model(service)
    if st is None:
        return model

    st.title("Ingestion Manager")
    st.subheader("Upload")
    uploaded_file = st.file_uploader("PDF file", type=["pdf"])
    collection = st.text_input("Collection", value="default")
    force = st.checkbox("Force re-ingest", value=False)

    if st.button("Run ingestion", disabled=uploaded_file is None):
        progress_bar = st.progress(0)
        status = st.empty()

        def on_progress(stage: str, current: int, total: int) -> None:
            progress_bar.progress(current / total if total else 1.0)
            status.write(f"{stage} ({current}/{total})")

        try:
            result = service.ingest_upload(
                uploaded_file,
                collection=collection.strip() or "default",
                force=force,
                progress_callback=on_progress,
            )
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")
        else:
            st.success(
                f"Ingested {result.source_path}: {result.chunk_count} chunks, "
                f"{result.image_count} images, skipped={result.skipped}"
            )
            model = ingestion_manager_model(service)

    st.subheader("Documents")
    if not model["document_rows"]:
        st.info("No ingested documents found.")
        return model

    st.dataframe(model["document_rows"], use_container_width=True, hide_index=True)
    selected_source = st.selectbox("Document", [row["source_path"] for row in model["document_rows"]])
    selected = next((document for document in model["documents"] if document["source_path"] == selected_source), None)
    if selected is not None and st.button("Delete document"):
        try:
            deleted = service.delete_document(
                str(selected["source_path"]),
                collection=selected.get("collection"),
            )
        except Exception as exc:
            st.error(f"Delete failed: {exc}")
        else:
            st.success(
                "Deleted "
                f"{deleted['vector_deleted']} vectors, {deleted['bm25_deleted']} BM25 records, "
                f"{deleted['image_deleted']} images."
            )
            model = ingestion_manager_model(service)
    return model


def _document_row(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_path": document.get("source_path"),
        "collection": document.get("collection"),
        "chunks": document.get("chunk_count"),
        "images": document.get("image_count"),
        "processed_at": document.get("processed_at"),
    }


__all__ = ["ingestion_manager_model", "render"]
