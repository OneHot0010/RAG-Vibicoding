"""Offline ingestion command for building local RAG indexes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.settings import Settings, load_settings
from ingestion import IngestionPipeline, IngestionPipelineError
from ingestion.chunking import DocumentChunker
from ingestion.embedding import BatchProcessor, DenseEncoder, SparseEncoder
from ingestion.storage import BM25Indexer, ImageStorage, VectorUpserter
from libs.embedding import BaseEmbedding
from libs.loader import PdfLoader, SQLiteIntegrityChecker


class LocalHashEmbedding(BaseEmbedding):
    """Small deterministic embedding backend for offline ingestion smoke runs."""

    dimension = 8

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([round((digest[index] / 255.0), 6) for index in range(self.dimension)])
        return vectors


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.config)
        paths = _resolve_paths(args.path)
        pipeline = build_pipeline(settings, args.data_dir, offline_embedding=not args.online_embedding)
        results = [
            pipeline.run(path, collection=args.collection, force=args.force).to_dict()
            for path in paths
        ]
    except (IngestionPipelineError, ValueError, OSError) as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1

    skipped = sum(1 for result in results if result["skipped"])
    ingested = len(results) - skipped
    print(
        json.dumps(
            {
                "collection": args.collection,
                "ingested": ingested,
                "skipped": skipped,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest PDFs into local vector, BM25, and image indexes.")
    parser.add_argument("--path", required=True, help="PDF file or directory containing PDF files.")
    parser.add_argument("--collection", default="default", help="Target collection name.")
    parser.add_argument("--force", action="store_true", help="Re-ingest files even if their hash was already processed.")
    parser.add_argument("--config", default="config/settings.yaml", help="Settings YAML path.")
    parser.add_argument("--data-dir", default="data", help="Root directory for generated local indexes.")
    parser.add_argument(
        "--online-embedding",
        action="store_true",
        help="Use the embedding provider from settings instead of the offline deterministic backend.",
    )
    return parser


def build_pipeline(settings: Settings, data_dir: str | Path, offline_embedding: bool = True) -> IngestionPipeline:
    data_root = Path(data_dir)
    db_root = data_root / "db"
    settings = _with_vector_store_path(settings, db_root / "chroma")
    dense_encoder = DenseEncoder(settings, embedding=LocalHashEmbedding()) if offline_embedding else DenseEncoder(settings)
    return IngestionPipeline(
        settings=settings,
        integrity_checker=SQLiteIntegrityChecker(db_root / "ingestion_history.db"),
        loader=PdfLoader(images_root=data_root / "images" / "_extracted"),
        chunker=DocumentChunker(settings),
        batch_processor=BatchProcessor(
            settings,
            dense_encoder=dense_encoder,
            sparse_encoder=SparseEncoder(),
        ),
        bm25_indexer=BM25Indexer(index_dir=db_root / "bm25"),
        vector_upserter=VectorUpserter(settings),
        image_storage=ImageStorage(images_root=data_root / "images", db_path=db_root / "image_index.db"),
    )


def _resolve_paths(raw_path: str) -> list[Path]:
    path = Path(raw_path)
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"--path must point to a .pdf file or directory: {path}")
        return [path]
    if path.is_dir():
        paths = sorted(item for item in path.rglob("*.pdf") if item.is_file())
        if not paths:
            raise ValueError(f"no PDF files found under: {path}")
        return paths
    raise ValueError(f"path not found: {path}")


def _with_vector_store_path(settings: Settings, persist_path: Path) -> Settings:
    return replace(settings, vector_store=replace(settings.vector_store, persist_path=str(persist_path)))


if __name__ == "__main__":
    raise SystemExit(main())
