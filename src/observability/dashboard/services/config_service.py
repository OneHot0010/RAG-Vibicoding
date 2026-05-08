"""Configuration and asset summary service for the dashboard."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.settings import Settings, load_settings


@dataclass(frozen=True)
class ComponentInfo:
    """Display-ready configuration for one pluggable component."""

    name: str
    provider: str
    model: str | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize component information for tests and UI adapters."""
        return {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "details": self.details or {},
        }


@dataclass(frozen=True)
class DataAssetStats:
    """Local data asset counts shown on the overview page."""

    document_count: int
    chunk_count: int
    image_count: int
    trace_count: int
    data_dir: str
    trace_log_file: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize data asset statistics."""
        return {
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "image_count": self.image_count,
            "trace_count": self.trace_count,
            "data_dir": self.data_dir,
            "trace_log_file": self.trace_log_file,
        }


class ConfigService:
    """Load settings and prepare dashboard overview data."""

    def __init__(self, settings_path: str | Path = "config/settings.yaml", data_dir: str | Path = "data") -> None:
        self.settings_path = Path(settings_path)
        self.data_dir = Path(data_dir)

    def load(self) -> Settings:
        """Load project settings."""
        return load_settings(self.settings_path)

    def component_cards(self, settings: Settings | None = None) -> list[dict[str, Any]]:
        """Return display-ready component configuration cards."""
        settings = settings or self.load()
        cards = [
            ComponentInfo("LLM", settings.llm.provider, settings.llm.model),
            ComponentInfo("Embedding", settings.embedding.provider, settings.embedding.model),
            ComponentInfo(
                "Splitter",
                settings.splitter.strategy,
                details={"chunk_size": settings.splitter.chunk_size, "chunk_overlap": settings.splitter.chunk_overlap},
            ),
            ComponentInfo("Vector Store", settings.vector_store.backend, details={"persist_path": settings.vector_store.persist_path}),
            ComponentInfo(
                "Retrieval",
                settings.retrieval.sparse_backend,
                settings.retrieval.fusion_algorithm,
                {
                    "top_k_dense": settings.retrieval.top_k_dense,
                    "top_k_sparse": settings.retrieval.top_k_sparse,
                    "top_k_final": settings.retrieval.top_k_final,
                },
            ),
            ComponentInfo("Reranker", settings.rerank.backend, settings.rerank.model, {"top_m": settings.rerank.top_m}),
            ComponentInfo("Vision LLM", settings.vision_llm.provider, settings.vision_llm.model),
        ]
        return [card.to_dict() for card in cards]

    def data_asset_stats(self, settings: Settings | None = None) -> dict[str, Any]:
        """Return local document/chunk/image/trace counts."""
        settings = settings or self.load()
        trace_log = Path(settings.observability.log_file)
        if not trace_log.is_absolute():
            trace_log = Path.cwd() / trace_log
        stats = DataAssetStats(
            document_count=_count_documents(self.data_dir / "documents"),
            chunk_count=_count_chroma_records(self.data_dir / "db" / "chroma" / "records.json"),
            image_count=_count_images(self.data_dir / "db" / "image_index.db"),
            trace_count=_count_jsonl_lines(trace_log),
            data_dir=str(self.data_dir),
            trace_log_file=str(trace_log),
        )
        return stats.to_dict()

    def overview(self) -> dict[str, Any]:
        """Return all data needed for the overview page."""
        settings = self.load()
        return {
            "components": self.component_cards(settings),
            "assets": self.data_asset_stats(settings),
            "dashboard": {
                "enabled": settings.dashboard.enabled if settings.dashboard else False,
                "port": settings.dashboard.port if settings.dashboard else None,
                "auto_refresh": settings.dashboard.auto_refresh if settings.dashboard else False,
                "refresh_interval": settings.dashboard.refresh_interval if settings.dashboard else None,
            },
        }


def _count_documents(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.name != ".gitkeep")


def _count_chroma_records(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    return len(payload) if isinstance(payload, list) else 0


def _count_images(db_path: Path) -> int:
    if not db_path.is_file():
        return 0
    try:
        with sqlite3.connect(db_path) as connection:
            row = connection.execute("SELECT COUNT(*) FROM image_index").fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0]) if row else 0


def _count_jsonl_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
