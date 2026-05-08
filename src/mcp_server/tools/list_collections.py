"""Collection listing MCP tool."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp_server.protocol_handler import INVALID_PARAMS, ProtocolError, ToolSpec


@dataclass
class CollectionStats:
    """Aggregated statistics for one knowledge collection."""

    name: str
    description: str = "Indexed collection"
    document_count: int = 0
    chunk_count: int = 0
    image_count: int = 0
    paths: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        """Serialize collection stats for MCP structuredContent."""
        return {
            "name": self.name,
            "description": self.description,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "image_count": self.image_count,
            "paths": sorted(self.paths),
            "sources": sorted(self.sources),
        }


def list_collections(arguments: dict[str, Any]) -> dict[str, Any]:
    """List local knowledge collections and lightweight statistics."""
    data_dir = arguments.get("data_dir", "data")
    if not isinstance(data_dir, str) or not data_dir.strip():
        raise ProtocolError(INVALID_PARAMS, "data_dir must be a non-empty string")

    root = Path(data_dir)
    collections = _collect_stats(root)
    collection_dicts = [stats.to_dict() for stats in collections]
    return {
        "content": [{"type": "text", "text": _render_markdown(collection_dicts)}],
        "structuredContent": {
            "collections": collection_dicts,
            "total_collections": len(collection_dicts),
            "data_dir": str(root),
        },
    }


def list_collections_tool_spec() -> ToolSpec:
    """Return MCP registration metadata for list_collections."""
    return ToolSpec(
        name="list_collections",
        description="List local knowledge hub collections with document, chunk, and image counts.",
        input_schema={
            "type": "object",
            "properties": {},
        },
        handler=list_collections,
    )


def _collect_stats(data_dir: Path) -> list[CollectionStats]:
    stats: dict[str, CollectionStats] = {}
    _add_document_collections(data_dir / "documents", stats)
    _add_vector_record_stats(data_dir / "db" / "chroma" / "records.json", stats)
    _add_image_stats(data_dir / "db" / "image_index.db", stats)
    return [stats[name] for name in sorted(stats)]


def _stats_for(stats: dict[str, CollectionStats], name: str) -> CollectionStats:
    normalized = name.strip() or "default"
    if normalized not in stats:
        stats[normalized] = CollectionStats(name=normalized)
    return stats[normalized]


def _add_document_collections(documents_root: Path, stats: dict[str, CollectionStats]) -> None:
    if not documents_root.is_dir():
        return

    root_files = _document_files(documents_root, recursive=False)
    if root_files:
        item = _stats_for(stats, "default")
        item.description = f"Documents under {documents_root}"
        item.document_count += len(root_files)
        item.paths.add(str(documents_root))
        item.sources.update(str(path) for path in root_files)

    for child in sorted(path for path in documents_root.iterdir() if path.is_dir()):
        files = _document_files(child, recursive=True)
        item = _stats_for(stats, child.name)
        item.description = f"Documents under {child}"
        item.document_count += len(files)
        item.paths.add(str(child))
        item.sources.update(str(path) for path in files)


def _document_files(root: Path, recursive: bool) -> list[Path]:
    candidates = root.rglob("*") if recursive else root.iterdir()
    return sorted(path for path in candidates if path.is_file() and path.name != ".gitkeep")


def _add_vector_record_stats(records_path: Path, stats: dict[str, CollectionStats]) -> None:
    if not records_path.is_file():
        return
    try:
        payload = json.loads(records_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if not isinstance(payload, list):
        return

    sources_by_collection: dict[str, set[str]] = {}
    chunks_by_collection: dict[str, int] = {}
    for record in payload:
        if not isinstance(record, dict):
            continue
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        collection = str(metadata.get("collection") or "default")
        chunks_by_collection[collection] = chunks_by_collection.get(collection, 0) + 1
        source = metadata.get("source_path") or metadata.get("source")
        if source:
            sources_by_collection.setdefault(collection, set()).add(str(source))

    for name, chunk_count in chunks_by_collection.items():
        item = _stats_for(stats, name)
        item.chunk_count += chunk_count
        sources = sources_by_collection.get(name, set())
        item.sources.update(sources)
        if sources and item.document_count == 0:
            item.document_count = len(sources)


def _add_image_stats(db_path: Path, stats: dict[str, CollectionStats]) -> None:
    if not db_path.is_file():
        return
    try:
        with sqlite3.connect(db_path) as connection:
            rows = connection.execute(
                "SELECT collection, COUNT(*) AS image_count FROM image_index GROUP BY collection"
            ).fetchall()
    except sqlite3.Error:
        return
    for collection, image_count in rows:
        item = _stats_for(stats, str(collection or "default"))
        item.image_count += int(image_count)


def _render_markdown(collections: list[dict[str, Any]]) -> str:
    if not collections:
        return "未找到知识库集合，请先运行 ingest.py 摄取数据或在 data/documents/ 下添加文档。"

    lines = [
        "## Knowledge Hub Collections",
        "",
        "| Collection | Documents | Chunks | Images |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in collections:
        lines.append(
            f"| {item['name']} | {item['document_count']} | {item['chunk_count']} | {item['image_count']} |"
        )
    return "\n".join(lines)
