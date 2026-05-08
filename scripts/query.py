"""Developer CLI for querying the local knowledge hub indexes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.query_engine import CoreReranker, DenseRetriever, HybridSearch, QueryProcessor, RRFusion, SparseRetriever
from core.settings import Settings, load_settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from ingestion.storage import BM25Indexer
from libs.embedding import BaseEmbedding
from libs.vector_store import VectorStoreFactory


NO_DATA_MESSAGE = "未找到相关文档，请先运行 ingest.py 摄取数据。"


class LocalHashEmbedding(BaseEmbedding):
    """Deterministic embedding backend matching scripts/ingest.py offline mode."""

    dimension = 8

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([round((digest[index] / 255.0), 6) for index in range(self.dimension)])
        return vectors


@dataclass(frozen=True)
class QueryComponents:
    """Runtime query components for the CLI."""

    settings: Settings
    hybrid_search: HybridSearch
    reranker: CoreReranker


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    filters = {"collection": args.collection} if args.collection else {}
    trace = TraceContext()

    try:
        settings = load_settings(args.config)
        components = build_query_components(settings, args.data_dir, offline_embedding=not args.online_embedding)
        fused_results = components.hybrid_search.search(args.query, top_k=args.top_k, filters=filters, trace=trace)
        results = fused_results if args.no_rerank else components.reranker.rerank(
            args.query, fused_results, top_k=args.top_k, trace=trace
        )
    except FileNotFoundError:
        print(NO_DATA_MESSAGE)
        return 0
    except Exception as exc:
        print(f"query failed: {exc}", file=sys.stderr)
        return 1

    if not results:
        print(NO_DATA_MESSAGE)
        if args.verbose:
            print(_format_verbose(trace, [], []))
        return 0

    print(_format_results(results))
    if args.verbose:
        print(_format_verbose(trace, fused_results, results))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query local RAG indexes built by scripts/ingest.py.")
    parser.add_argument("--query", required=True, help="User query text.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return.")
    parser.add_argument("--collection", help="Optional collection metadata filter.")
    parser.add_argument("--verbose", action="store_true", help="Show trace and intermediate debug details.")
    parser.add_argument("--no-rerank", action="store_true", help="Skip reranker stage and return fused results.")
    parser.add_argument("--config", default="config/settings.yaml", help="Settings YAML path.")
    parser.add_argument("--data-dir", default="data", help="Root directory containing local indexes.")
    parser.add_argument(
        "--online-embedding",
        action="store_true",
        help="Use the embedding provider from settings instead of the offline deterministic backend.",
    )
    return parser


def build_query_components(
    settings: Settings,
    data_dir: str | Path,
    offline_embedding: bool = True,
) -> QueryComponents:
    data_root = Path(data_dir)
    db_root = data_root / "db"
    chroma_dir = db_root / "chroma"
    bm25_dir = db_root / "bm25"
    _ensure_index_exists(chroma_dir, bm25_dir)

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
    return QueryComponents(settings=settings, hybrid_search=hybrid, reranker=CoreReranker(settings))


def _ensure_index_exists(chroma_dir: Path, bm25_dir: Path) -> None:
    if not (chroma_dir / "records.json").is_file() or not (bm25_dir / "index.json").is_file():
        raise FileNotFoundError(NO_DATA_MESSAGE)


def _format_results(results: list[RetrievalResult]) -> str:
    lines = ["Top-K 检索结果", ""]
    for index, result in enumerate(results, start=1):
        source = result.metadata.get("source_path", "unknown")
        page = result.metadata.get("page") or result.metadata.get("page_num") or "-"
        lines.extend(
            [
                f"{index}. score={result.score:.6f} chunk_id={result.chunk_id}",
                f"   source={source} page={page}",
                f"   {_summarize(result.text)}",
            ]
        )
    return "\n".join(lines)


def _format_verbose(
    trace: TraceContext,
    fused_results: list[RetrievalResult],
    final_results: list[RetrievalResult],
) -> str:
    payload = {
        "trace": trace.stages,
        "fusion_results": [result.to_dict() for result in fused_results],
        "rerank_results": [result.to_dict() for result in final_results],
    }
    return "\nVerbose\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _summarize(text: str, limit: int = 180) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return f"{clean[: limit - 3]}..."


if __name__ == "__main__":
    raise SystemExit(main())
