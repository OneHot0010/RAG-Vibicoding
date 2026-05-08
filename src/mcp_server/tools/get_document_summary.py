"""Document summary MCP tool."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from mcp_server.protocol_handler import INVALID_PARAMS, ProtocolError, ToolSpec


def get_document_summary(arguments: dict[str, Any]) -> dict[str, Any]:
    """Return title, summary, tags, and source metadata for one indexed document."""
    doc_id = arguments.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ProtocolError(INVALID_PARAMS, "doc_id must be a non-empty string")

    data_dir = arguments.get("data_dir", "data")
    if not isinstance(data_dir, str) or not data_dir.strip():
        raise ProtocolError(INVALID_PARAMS, "data_dir must be a non-empty string")

    collection = arguments.get("collection")
    if collection is not None and (not isinstance(collection, str) or not collection.strip()):
        raise ProtocolError(INVALID_PARAMS, "collection must be a non-empty string when provided")

    root = Path(data_dir)
    records = _matching_records(root / "db" / "chroma" / "records.json", doc_id.strip(), collection)
    document = _build_document_summary(root, doc_id.strip(), records)
    return {
        "content": [{"type": "text", "text": _render_markdown(doc_id.strip(), document)}],
        "structuredContent": {
            "doc_id": doc_id.strip(),
            "found": document is not None,
            "document": document,
            "data_dir": str(root),
        },
    }


def get_document_summary_tool_spec() -> ToolSpec:
    """Return MCP registration metadata for get_document_summary."""
    return ToolSpec(
        name="get_document_summary",
        description="Return title, summary, tags, and source metadata for an indexed document.",
        input_schema={
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "Document id, source path, file name, file stem, or file hash.",
                },
                "collection": {"type": "string", "description": "Optional collection filter."},
            },
            "required": ["doc_id"],
        },
        handler=get_document_summary,
    )


def _matching_records(records_path: Path, doc_id: str, collection: str | None) -> list[dict[str, Any]]:
    if not records_path.is_file():
        return []
    try:
        payload = json.loads(records_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []

    matched: list[dict[str, Any]] = []
    for record in payload:
        if not isinstance(record, dict):
            continue
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        if collection is not None and metadata.get("collection") != collection.strip():
            continue
        if _record_matches_doc_id(record, metadata, doc_id):
            matched.append(record)
    return sorted(matched, key=_record_sort_key)


def _record_matches_doc_id(record: dict[str, Any], metadata: dict[str, Any], doc_id: str) -> bool:
    normalized = doc_id.strip()
    candidates = {
        str(record.get("id") or ""),
        str(metadata.get("doc_id") or ""),
        str(metadata.get("document_id") or ""),
        str(metadata.get("file_hash") or ""),
        str(metadata.get("content_hash") or ""),
        str(metadata.get("original_chunk_id") or ""),
    }
    source = metadata.get("source_path") or metadata.get("source")
    if source:
        source_path = Path(str(source))
        candidates.update({str(source), source_path.name, source_path.stem})
    return normalized in {candidate for candidate in candidates if candidate}


def _record_sort_key(record: dict[str, Any]) -> tuple[int, str]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    chunk_index = metadata.get("chunk_index")
    try:
        index = int(chunk_index)
    except (TypeError, ValueError):
        index = 0
    return index, str(record.get("id") or "")


def _build_document_summary(data_dir: Path, doc_id: str, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None

    metadata_items = [
        record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        for record in records
    ]
    source = _first_text(metadata_items, "source_path") or _first_text(metadata_items, "source")
    file_hash = _first_text(metadata_items, "file_hash")
    title = _first_text(metadata_items, "title") or (Path(source).stem if source else doc_id)
    summary = _combine_summaries(records)
    tags = _combine_tags(metadata_items)
    pages = sorted(
        {
            int(metadata["page"])
            for metadata in metadata_items
            if isinstance(metadata.get("page"), int)
        }
    )
    return {
        "doc_id": doc_id,
        "title": title,
        "summary": summary,
        "tags": tags,
        "source": source,
        "collection": _first_text(metadata_items, "collection"),
        "created_at": _lookup_processed_at(data_dir / "db" / "ingestion_history.db", file_hash, source),
        "chunk_count": len(records),
        "chunk_ids": [str(record.get("id") or "") for record in records],
        "pages": pages,
    }


def _first_text(metadata_items: list[dict[str, Any]], key: str) -> str | None:
    for metadata in metadata_items:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _combine_summaries(records: list[dict[str, Any]]) -> str:
    summaries: list[str] = []
    for record in records:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        summary = metadata.get("summary")
        if isinstance(summary, str) and summary.strip() and summary.strip() not in summaries:
            summaries.append(summary.strip())
    if summaries:
        return " ".join(summaries)

    text = " ".join(str(record.get("text") or "").strip() for record in records).strip()
    words = text.split()
    if not words:
        return "No summary available."
    return " ".join(words[:80])


def _combine_tags(metadata_items: list[dict[str, Any]]) -> list[str]:
    counter: Counter[str] = Counter()
    for metadata in metadata_items:
        tags = metadata.get("tags")
        if not isinstance(tags, list):
            continue
        for tag in tags:
            if isinstance(tag, str) and tag.strip():
                counter[tag.strip()] += 1
    return [tag for tag, _ in counter.most_common(10)]


def _lookup_processed_at(db_path: Path, file_hash: str | None, source: str | None) -> str | None:
    if not db_path.is_file() or (not file_hash and not source):
        return None
    clauses: list[str] = []
    params: list[str] = []
    if file_hash:
        clauses.append("file_hash = ?")
        params.append(file_hash)
    if source:
        clauses.append("file_path = ?")
        params.append(source)
    try:
        with sqlite3.connect(db_path) as connection:
            row = connection.execute(
                f"SELECT processed_at FROM ingestion_history WHERE {' OR '.join(clauses)} ORDER BY processed_at DESC LIMIT 1",
                params,
            ).fetchone()
    except sqlite3.Error:
        return None
    return str(row[0]) if row else None


def _render_markdown(doc_id: str, document: dict[str, Any] | None) -> str:
    if document is None:
        return f"Document not found: `{doc_id}`. Run ingest.py first or check the document id/source name."

    tags = ", ".join(document["tags"]) if document["tags"] else "none"
    source = document["source"] or "unknown"
    created_at = document["created_at"] or "unknown"
    return "\n".join(
        [
            f"## {document['title']}",
            "",
            document["summary"],
            "",
            f"- Source: `{source}`",
            f"- Collection: `{document['collection'] or 'default'}`",
            f"- Chunks: {document['chunk_count']}",
            f"- Created At: {created_at}",
            f"- Tags: {tags}",
        ]
    )
