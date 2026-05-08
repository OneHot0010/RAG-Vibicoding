"""Knowledge hub query MCP tool."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

from core.query_engine import CoreReranker, DenseRetriever, HybridSearch, QueryProcessor, RRFusion, SparseRetriever
from core.response import ResponseBuilder
from core.settings import Settings, load_settings
from core.trace.trace_context import TraceContext
from ingestion.storage import BM25Indexer
from libs.embedding import BaseEmbedding
from libs.vector_store import VectorStoreFactory
from mcp_server.protocol_handler import INVALID_PARAMS, ProtocolError, ToolSpec


class LocalHashEmbedding(BaseEmbedding):
    """Deterministic embedding backend matching the offline CLI tools."""

    dimension = 8

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([round((digest[index] / 255.0), 6) for index in range(self.dimension)])
        return vectors


def query_knowledge_hub(arguments: dict[str, Any]) -> dict[str, Any]:
    """Query the local knowledge hub and return MCP content plus citations."""
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ProtocolError(INVALID_PARAMS, "query must be a non-empty string")
    top_k = _top_k(arguments.get("top_k", 5))
    collection = arguments.get("collection")
    if collection is not None and (not isinstance(collection, str) or not collection.strip()):
        raise ProtocolError(INVALID_PARAMS, "collection must be a non-empty string when provided")

    config = str(arguments.get("config", "config/settings.yaml"))
    data_dir = Path(str(arguments.get("data_dir", "data")))
    no_rerank = bool(arguments.get("no_rerank", False))
    online_embedding = bool(arguments.get("online_embedding", False))

    try:
        settings = load_settings(config)
        components = _build_components(settings, data_dir, offline_embedding=not online_embedding)
        filters = {"collection": collection.strip()} if isinstance(collection, str) else {}
        trace = TraceContext()
        fused = components["hybrid"].search(query, top_k=top_k, filters=filters, trace=trace)
        results = fused if no_rerank else components["reranker"].rerank(query, fused, top_k=top_k, trace=trace)
    except FileNotFoundError:
        results = []

    return ResponseBuilder().build(results, query).to_dict()


def query_knowledge_hub_tool_spec() -> ToolSpec:
    """Return MCP registration metadata for query_knowledge_hub."""
    return ToolSpec(
        name="query_knowledge_hub",
        description="Search the local RAG knowledge hub and return cited Markdown results.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "User query text."},
                "top_k": {"type": "integer", "minimum": 1, "default": 5},
                "collection": {"type": "string", "description": "Optional collection filter."},
            },
            "required": ["query"],
        },
        handler=query_knowledge_hub,
    )


def _build_components(settings: Settings, data_dir: Path, offline_embedding: bool = True) -> dict[str, Any]:
    db_root = data_dir / "db"
    chroma_dir = db_root / "chroma"
    bm25_dir = db_root / "bm25"
    if not (chroma_dir / "records.json").is_file() or not (bm25_dir / "index.json").is_file():
        raise FileNotFoundError("knowledge hub indexes not found")

    settings = replace(settings, vector_store=replace(settings.vector_store, persist_path=str(chroma_dir)))
    vector_store = VectorStoreFactory.create(settings)
    embedding = LocalHashEmbedding() if offline_embedding else None
    dense = DenseRetriever(settings, embedding_client=embedding, vector_store=vector_store)
    sparse = SparseRetriever(settings, bm25_indexer=BM25Indexer.load(bm25_dir), vector_store=vector_store)
    hybrid = HybridSearch(
        settings,
        query_processor=QueryProcessor(),
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion=RRFusion(),
    )
    return {"hybrid": hybrid, "reranker": CoreReranker(settings)}


def _top_k(value: Any) -> int:
    try:
        top_k = int(value)
    except (TypeError, ValueError):
        raise ProtocolError(INVALID_PARAMS, "top_k must be an integer") from None
    if top_k <= 0:
        raise ProtocolError(INVALID_PARAMS, "top_k must be greater than 0")
    return top_k
