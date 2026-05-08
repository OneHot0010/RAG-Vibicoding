"""Tests for dashboard overview services and app registry."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from observability.dashboard.app import page_registry
from observability.dashboard.pages.overview import overview_model, render
from observability.dashboard.services.config_service import ConfigService


def write_settings(path: Path, trace_log: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
llm:
  provider: fake
  model: fake-chat
embedding:
  provider: fake
  model: fake-embedding
vision_llm:
  provider: fake
  model: fake-vision
splitter:
  strategy: recursive
  chunk_size: 100
  chunk_overlap: 10
vector_store:
  backend: chroma
  persist_path: ./data/db/chroma
retrieval:
  sparse_backend: bm25
  fusion_algorithm: rrf
  top_k_dense: 4
  top_k_sparse: 5
  top_k_final: 3
rerank:
  backend: none
  model: fake-reranker
  top_m: 2
evaluation:
  backends:
    - custom
  golden_test_set: ./golden.json
observability:
  enabled: true
  log_file: {trace_log.as_posix()}
dashboard:
  enabled: true
  port: 8501
  traces_dir: ./logs
  auto_refresh: true
  refresh_interval: 5
""".strip(),
        encoding="utf-8",
    )


def seed_assets(data_dir: Path, trace_log: Path) -> None:
    docs = data_dir / "documents" / "docs"
    docs.mkdir(parents=True)
    (docs / "guide.md").write_text("guide", encoding="utf-8")
    chroma = data_dir / "db" / "chroma"
    chroma.mkdir(parents=True)
    (chroma / "records.json").write_text(
        json.dumps([{"id": "a"}, {"id": "b"}]),
        encoding="utf-8",
    )
    image_db = data_dir / "db" / "image_index.db"
    with sqlite3.connect(image_db) as connection:
        connection.execute("CREATE TABLE image_index (image_id TEXT)")
        connection.executemany("INSERT INTO image_index (image_id) VALUES (?)", [("img-1",), ("img-2",)])
    trace_log.parent.mkdir(parents=True)
    trace_log.write_text('{"trace_id":"1"}\n{"trace_id":"2"}\n', encoding="utf-8")


def test_config_service_formats_components_and_asset_stats(tmp_path: Path) -> None:
    settings_path = tmp_path / "config" / "settings.yaml"
    data_dir = tmp_path / "data"
    trace_log = tmp_path / "logs" / "traces.jsonl"
    write_settings(settings_path, trace_log)
    seed_assets(data_dir, trace_log)

    service = ConfigService(settings_path=settings_path, data_dir=data_dir)
    overview = service.overview()

    assert [component["name"] for component in overview["components"]] == [
        "LLM",
        "Embedding",
        "Splitter",
        "Vector Store",
        "Retrieval",
        "Reranker",
        "Vision LLM",
    ]
    assert overview["components"][0]["provider"] == "fake"
    assert overview["assets"]["document_count"] == 1
    assert overview["assets"]["chunk_count"] == 2
    assert overview["assets"]["image_count"] == 2
    assert overview["assets"]["trace_count"] == 2
    assert overview["dashboard"]["port"] == 8501


def test_overview_model_and_render_return_same_data_without_streamlit(tmp_path: Path) -> None:
    settings_path = tmp_path / "config" / "settings.yaml"
    trace_log = tmp_path / "logs" / "traces.jsonl"
    write_settings(settings_path, trace_log)

    model = overview_model(settings_path=str(settings_path), data_dir=str(tmp_path / "data"))
    rendered = render(settings_path=str(settings_path), data_dir=str(tmp_path / "data"))

    assert rendered == model
    assert model["assets"]["document_count"] == 0


def test_dashboard_page_registry_has_expected_pages() -> None:
    assert list(page_registry()) == [
        "Overview",
        "Data Browser",
        "Ingestion Manager",
        "Ingestion Traces",
        "Query Traces",
        "Evaluation",
    ]
