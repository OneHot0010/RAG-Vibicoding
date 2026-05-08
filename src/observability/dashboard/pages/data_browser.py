"""Data browser page for ingested documents, chunks, and images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from observability.dashboard.services.data_service import DataService


ALL_COLLECTIONS = "All collections"


def data_browser_model(
    data_service: DataService | None = None,
    *,
    data_dir: str | Path = "data",
    collection: str | None = None,
    search: str = "",
    selected_doc_id: str | None = None,
) -> dict[str, Any]:
    """Return display data for the data browser page."""
    service = data_service or DataService(data_dir=data_dir)
    all_documents = service.list_documents()
    collections = sorted({str(document["collection"]) for document in all_documents if document.get("collection")})

    normalized_collection = _normalize_collection(collection)
    documents = service.list_documents(collection=normalized_collection)
    documents = _filter_documents(documents, search)

    selected_document = _select_document(documents, selected_doc_id)
    detail: dict[str, Any] | None = None
    detail_error: str | None = None
    if selected_document is not None:
        doc_id = str(selected_document["source_path"])
        try:
            detail = service.get_document_detail(doc_id)
        except ValueError as exc:
            detail_error = str(exc)

    return {
        "collections": collections,
        "selected_collection": normalized_collection,
        "search": search.strip(),
        "documents": documents,
        "selected_document": selected_document,
        "detail": detail,
        "detail_error": detail_error,
        "document_rows": [_document_row(document) for document in documents],
        "chunk_rows": _chunk_rows(detail),
        "image_rows": _image_rows(detail),
    }


def render(
    st: Any | None = None,
    *,
    data_service: DataService | None = None,
    data_dir: str | Path = "data",
) -> dict[str, Any]:
    """Render the data browser page and return the underlying model for tests."""
    if st is None:
        return data_browser_model(data_service=data_service, data_dir=data_dir)

    service = data_service or DataService(data_dir=data_dir)
    initial = data_browser_model(data_service=service)
    collection_options = [ALL_COLLECTIONS, *initial["collections"]]

    st.title("Data Browser")
    controls = st.columns([1, 2])
    selected_collection = controls[0].selectbox("Collection", collection_options, index=0)
    search = controls[1].text_input("Search", value="")

    model = data_browser_model(
        data_service=service,
        collection=selected_collection,
        search=search,
    )

    summary = st.columns(3)
    summary[0].metric("Documents", len(model["documents"]))
    summary[1].metric("Chunks", sum(int(document.get("chunk_count") or 0) for document in model["documents"]))
    summary[2].metric("Images", sum(int(document.get("image_count") or 0) for document in model["documents"]))

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Documents")
        if model["document_rows"]:
            st.dataframe(model["document_rows"], use_container_width=True, hide_index=True)
            selected_source = st.selectbox(
                "Open document",
                [row["source_path"] for row in model["document_rows"]],
                index=0,
            )
            model = data_browser_model(
                data_service=service,
                collection=selected_collection,
                search=search,
                selected_doc_id=selected_source,
            )
        else:
            st.info("No documents found.")

    with right:
        _render_detail(st, model)

    return model


def _render_detail(st: Any, model: dict[str, Any]) -> None:
    detail = model["detail"]
    if model["detail_error"]:
        st.error(model["detail_error"])
        return
    if detail is None:
        st.subheader("Document")
        st.info("Select a document.")
        return

    document = detail["document"]
    st.subheader(Path(str(document["source_path"])).name)
    stats = st.columns(3)
    stats[0].metric("Chunks", document["chunk_count"])
    stats[1].metric("Images", document["image_count"])
    stats[2].metric("Collection", document["collection"] or "-")
    st.write(
        {
            "source_path": document["source_path"],
            "doc_hash": document["doc_hash"],
            "processed_at": document["processed_at"],
        }
    )

    chunks_tab, images_tab = st.tabs(["Chunks", "Images"])
    with chunks_tab:
        if not detail["chunks"]:
            st.info("No chunks found.")
        for index, chunk in enumerate(detail["chunks"], start=1):
            label = f"{index}. {chunk['id']}"
            with st.expander(label, expanded=index == 1):
                st.text_area("Text", value=chunk["text"], height=160, disabled=True, key=f"chunk-text-{chunk['id']}")
                st.json(chunk["metadata"])

    with images_tab:
        if not detail["images"]:
            st.info("No images found.")
        for image in detail["images"]:
            path = Path(str(image["file_path"]))
            st.write(
                {
                    "image_id": image["image_id"],
                    "page_num": image["page_num"],
                    "created_at": image["created_at"],
                    "file_path": image["file_path"],
                }
            )
            if path.is_file():
                st.image(str(path), caption=image["image_id"], use_container_width=True)


def _normalize_collection(collection: str | None) -> str | None:
    if collection is None:
        return None
    value = collection.strip()
    return None if not value or value == ALL_COLLECTIONS else value


def _filter_documents(documents: list[dict[str, Any]], search: str) -> list[dict[str, Any]]:
    term = search.strip().lower()
    if not term:
        return documents
    return [document for document in documents if term in _document_search_text(document)]


def _document_search_text(document: dict[str, Any]) -> str:
    values = [
        document.get("source_path"),
        document.get("collection"),
        document.get("doc_hash"),
        document.get("processed_at"),
    ]
    return " ".join(str(value).lower() for value in values if value is not None)


def _select_document(documents: list[dict[str, Any]], selected_doc_id: str | None) -> dict[str, Any] | None:
    if not documents:
        return None
    if selected_doc_id:
        for document in documents:
            candidates = {
                str(document.get("source_path") or ""),
                str(document.get("doc_hash") or ""),
                Path(str(document.get("source_path") or "")).name,
                Path(str(document.get("source_path") or "")).stem,
            }
            if selected_doc_id in candidates:
                return document
    return documents[0]


def _document_row(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_path": document.get("source_path"),
        "collection": document.get("collection"),
        "chunks": document.get("chunk_count"),
        "images": document.get("image_count"),
        "processed_at": document.get("processed_at"),
    }


def _chunk_rows(detail: dict[str, Any] | None) -> list[dict[str, Any]]:
    if detail is None:
        return []
    rows = []
    for chunk in detail["chunks"]:
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        rows.append(
            {
                "id": chunk.get("id"),
                "chunk_index": metadata.get("chunk_index"),
                "text": chunk.get("text"),
                "metadata": metadata,
            }
        )
    return rows


def _image_rows(detail: dict[str, Any] | None) -> list[dict[str, Any]]:
    if detail is None:
        return []
    return [
        {
            "image_id": image.get("image_id"),
            "file_path": image.get("file_path"),
            "page_num": image.get("page_num"),
            "created_at": image.get("created_at"),
        }
        for image in detail["images"]
    ]


__all__ = ["ALL_COLLECTIONS", "data_browser_model", "render"]
