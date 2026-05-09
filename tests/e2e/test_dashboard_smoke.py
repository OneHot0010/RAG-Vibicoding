"""End-to-end smoke checks for Dashboard pages."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from observability.dashboard.pages import (
    data_browser,
    evaluation_panel,
    ingestion_manager,
    ingestion_traces,
    overview,
    query_traces,
)
from observability.dashboard.services.data_service import DataService
from observability.dashboard.services.evaluation_service import EvaluationService
from observability.dashboard.services.ingestion_service import IngestionService


class RecordingStreamlit:
    """Small Streamlit-like recorder for page-level smoke rendering."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.sidebar = self

    def __enter__(self) -> "RecordingStreamlit":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def columns(self, spec: int | list[int]) -> list["RecordingStreamlit"]:
        count = spec if isinstance(spec, int) else len(spec)
        self._record("columns", spec)
        return [self for _ in range(count)]

    def tabs(self, labels: list[str]) -> list["RecordingStreamlit"]:
        self._record("tabs", labels)
        return [self for _ in labels]

    def expander(self, label: str, expanded: bool = False) -> "RecordingStreamlit":
        self._record("expander", label, expanded=expanded)
        return self

    def empty(self) -> "RecordingStreamlit":
        self._record("empty")
        return self

    def progress(self, value: float) -> "RecordingStreamlit":
        self._record("progress", value)
        return self

    def selectbox(self, _label: str, options: list[Any], index: int = 0, **_kwargs: Any) -> Any:
        return options[index] if options else None

    def radio(self, _label: str, options: list[Any], index: int = 0, **_kwargs: Any) -> Any:
        return options[index] if options else None

    def text_input(self, _label: str, value: str = "", **_kwargs: Any) -> str:
        return value

    def checkbox(self, _label: str, value: bool = False, **_kwargs: Any) -> bool:
        return value

    def number_input(self, _label: str, value: int = 0, **_kwargs: Any) -> int:
        return value

    def file_uploader(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def button(self, *_args: Any, **_kwargs: Any) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        def recorder(*args: Any, **kwargs: Any) -> None:
            self._record(name, *args, **kwargs)

        return recorder

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))


def test_dashboard_pages_render_with_seeded_data(tmp_path: Path) -> None:
    paths = seed_dashboard_fixture(tmp_path)
    st = RecordingStreamlit()

    overview_model = overview.render(st, settings_path=paths["settings"], data_dir=paths["data_dir"])
    data_model = data_browser.render(st, data_service=DataService(data_dir=paths["data_dir"]))
    ingestion_model = ingestion_manager.render(
        st,
        ingestion_service=IngestionService(settings_path=paths["settings"], data_dir=paths["data_dir"]),
    )
    ingestion_trace_model = ingestion_traces.render(st, trace_log_file=paths["trace_log"])
    query_trace_model = query_traces.render(st, trace_log_file=paths["trace_log"])
    evaluation_model = evaluation_panel.render(
        st,
        evaluation_service=EvaluationService(settings_path=paths["settings"], data_dir=paths["data_dir"]),
    )

    assert overview_model["assets"]["document_count"] == 1
    assert data_model["detail"]["document"]["source_path"] == "docs/dashboard_smoke.pdf"
    assert ingestion_model["document_rows"][0]["source_path"] == "docs/dashboard_smoke.pdf"
    assert ingestion_trace_model["selected_summary"]["trace_type"] == "ingestion"
    assert query_trace_model["selected_summary"]["trace_type"] == "query"
    assert evaluation_model["test_set"]["case_count"] == 1
    assert {call[0] for call in st.calls} >= {"title", "dataframe", "metric"}


def test_dashboard_pages_smoke_with_streamlit_apptest(tmp_path: Path) -> None:
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    paths = seed_dashboard_fixture(tmp_path)
    smoke_app = tmp_path / "dashboard_smoke_app.py"
    smoke_app.write_text(
        f"""
import sys
from pathlib import Path

sys.path.insert(0, {str((Path(__file__).resolve().parents[2] / "src")).__repr__()})

import streamlit as st
from observability.dashboard.pages import data_browser, evaluation_panel, ingestion_manager, ingestion_traces, overview, query_traces
from observability.dashboard.services.data_service import DataService
from observability.dashboard.services.evaluation_service import EvaluationService
from observability.dashboard.services.ingestion_service import IngestionService

st.set_page_config(page_title="Dashboard Smoke", layout="wide")
tabs = st.tabs(["Overview", "Data Browser", "Ingestion Manager", "Ingestion Traces", "Query Traces", "Evaluation"])
with tabs[0]:
    overview.render(st, settings_path={paths["settings"]!r}, data_dir={paths["data_dir"]!r})
with tabs[1]:
    data_browser.render(st, data_service=DataService(data_dir={paths["data_dir"]!r}))
with tabs[2]:
    ingestion_manager.render(st, ingestion_service=IngestionService(settings_path={paths["settings"]!r}, data_dir={paths["data_dir"]!r}))
with tabs[3]:
    ingestion_traces.render(st, trace_log_file={paths["trace_log"]!r})
with tabs[4]:
    query_traces.render(st, trace_log_file={paths["trace_log"]!r})
with tabs[5]:
    evaluation_panel.render(st, evaluation_service=EvaluationService(settings_path={paths["settings"]!r}, data_dir={paths["data_dir"]!r}))
""".strip(),
        encoding="utf-8",
    )

    app = streamlit_testing.AppTest.from_file(str(smoke_app)).run(timeout=10)

    assert not app.exception
    assert any("System Overview" in title.value for title in app.title)
    assert any("Data Browser" in title.value for title in app.title)


def seed_dashboard_fixture(tmp_path: Path) -> dict[str, str]:
    data_dir = tmp_path / "data"
    trace_log = tmp_path / "logs" / "traces.jsonl"
    golden = tmp_path / "golden_test_set.json"
    settings = tmp_path / "config" / "settings.yaml"
    write_settings(settings, trace_log, golden)
    write_golden_test_set(golden)
    seed_vector_records(data_dir)
    seed_ingestion_history(data_dir)
    write_traces(trace_log)
    return {
        "data_dir": str(data_dir),
        "trace_log": str(trace_log),
        "golden": str(golden),
        "settings": str(settings),
    }


def write_settings(settings_path: Path, trace_log: Path, golden: Path) -> None:
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
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
  golden_test_set: {golden.as_posix()}
observability:
  enabled: true
  log_file: {trace_log.as_posix()}
dashboard:
  enabled: true
  port: 8501
  traces_dir: ./logs
  auto_refresh: false
  refresh_interval: 5
""".strip(),
        encoding="utf-8",
    )


def write_golden_test_set(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "test_cases": [
                    {
                        "query": "dashboard smoke",
                        "expected_chunk_ids": ["dashboard-chunk-1"],
                        "expected_sources": ["docs/dashboard_smoke.pdf"],
                        "expected_answer": "Dashboard smoke data is visible.",
                        "contexts": ["Dashboard smoke data is visible in the knowledge hub."],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def seed_vector_records(data_dir: Path) -> None:
    documents = data_dir / "documents" / "docs"
    documents.mkdir(parents=True, exist_ok=True)
    (documents / "dashboard_smoke.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")

    chroma = data_dir / "db" / "chroma"
    chroma.mkdir(parents=True, exist_ok=True)
    (chroma / "records.json").write_text(
        json.dumps(
            [
                {
                    "id": "dashboard-chunk-1",
                    "vector": [0.1, 0.2, 0.3],
                    "text": "Dashboard smoke data is visible in the knowledge hub.",
                    "metadata": {
                        "source_path": "docs/dashboard_smoke.pdf",
                        "collection": "docs",
                        "doc_hash": "dashboard-hash",
                        "chunk_index": 0,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )


def seed_ingestion_history(data_dir: Path) -> None:
    db_path = data_dir / "db" / "ingestion_history.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE ingestion_history (
                file_hash TEXT,
                file_path TEXT,
                status TEXT,
                processed_at TEXT,
                chunk_count INTEGER
            )
            """
        )
        connection.execute(
            """
            INSERT INTO ingestion_history (file_hash, file_path, status, processed_at, chunk_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("dashboard-hash", "docs/dashboard_smoke.pdf", "success", "2026-05-09T00:00:00Z", 1),
        )


def write_traces(trace_log: Path) -> None:
    trace_log.parent.mkdir(parents=True, exist_ok=True)
    traces = [
        {
            "trace_id": "ingestion-smoke",
            "trace_type": "ingestion",
            "started_at": "2026-05-09T00:00:00Z",
            "finished_at": "2026-05-09T00:00:01Z",
            "total_elapsed_ms": 1000.0,
            "stages": [
                {
                    "name": "ingestion.dashboard.start",
                    "elapsed_ms": 1.0,
                    "data": {"source_path": "docs/dashboard_smoke.pdf", "collection": "docs"},
                },
                {
                    "name": "ingestion.dashboard.completed",
                    "elapsed_ms": 2.0,
                    "data": {"chunk_count": 1, "image_count": 0, "collection": "docs"},
                },
            ],
        },
        {
            "trace_id": "query-smoke",
            "trace_type": "query",
            "started_at": "2026-05-09T00:00:02Z",
            "finished_at": "2026-05-09T00:00:03Z",
            "total_elapsed_ms": 750.0,
            "stages": [
                {
                    "name": "query.start",
                    "elapsed_ms": 1.0,
                    "data": {"query": "dashboard smoke", "filters": {"collection": "docs"}},
                },
                {
                    "name": "dense.retrieve.completed",
                    "elapsed_ms": 2.0,
                    "data": {
                        "dense_results": [
                            {
                                "id": "dashboard-chunk-1",
                                "score": 0.9,
                                "text": "Dashboard smoke data",
                                "metadata": {"source_path": "docs/dashboard_smoke.pdf"},
                            }
                        ]
                    },
                },
                {
                    "name": "query.completed",
                    "elapsed_ms": 3.0,
                    "data": {"result_count": 1, "rerank_enabled": False},
                },
            ],
        },
    ]
    trace_log.write_text("\n".join(json.dumps(trace) for trace in traces) + "\n", encoding="utf-8")
