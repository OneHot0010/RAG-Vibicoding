"""Tests for dashboard ingestion trace services and page model."""

from __future__ import annotations

import json
from pathlib import Path

from observability.dashboard.pages.ingestion_traces import ingestion_traces_model, render
from observability.dashboard.services.trace_service import TraceService


def write_trace_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "trace_id": "query-1",
            "trace_type": "query",
            "started_at": "2026-05-08T09:00:00",
            "finished_at": "2026-05-08T09:00:01",
            "total_elapsed_ms": 1.0,
            "stages": [],
        },
        {
            "trace_id": "ingest-old",
            "trace_type": "ingestion",
            "started_at": "2026-05-08T10:00:00",
            "finished_at": "2026-05-08T10:00:02",
            "total_elapsed_ms": 20.0,
            "stages": [
                {
                    "name": "pipeline.start",
                    "elapsed_ms": 1.0,
                    "started_at": "2026-05-08T10:00:00",
                    "data": {"source_path": "docs/old.pdf", "collection": "docs"},
                },
                {
                    "name": "pipeline.completed",
                    "elapsed_ms": 2.0,
                    "started_at": "2026-05-08T10:00:01",
                    "data": {"chunk_count": 1, "image_count": 0},
                },
            ],
        },
        {
            "trace_id": "ingest-new",
            "trace_type": "ingestion",
            "started_at": "2026-05-08T11:00:00",
            "finished_at": "2026-05-08T11:00:03",
            "total_elapsed_ms": 30.0,
            "stages": [
                {
                    "name": "ingestion.dashboard.start",
                    "elapsed_ms": 3.0,
                    "started_at": "2026-05-08T11:00:00",
                    "data": {"source_path": "docs/new.pdf", "collection": "docs", "force": True},
                },
                {
                    "name": "batch_processor.process",
                    "elapsed_ms": 8.0,
                    "started_at": "2026-05-08T11:00:01",
                    "data": {"chunk_count": 3, "provider": "local"},
                },
                {
                    "name": "image_storage.index",
                    "elapsed_ms": 4.0,
                    "started_at": "2026-05-08T11:00:02",
                    "data": {"count": 2},
                },
                {
                    "name": "ingestion.dashboard.completed",
                    "elapsed_ms": 1.0,
                    "started_at": "2026-05-08T11:00:03",
                    "data": {"chunk_count": 3, "image_count": 2, "skipped": False},
                },
            ],
        },
        {
            "trace_id": "ingest-failed",
            "trace_type": "ingestion",
            "started_at": "2026-05-08T08:00:00",
            "finished_at": "2026-05-08T08:00:01",
            "total_elapsed_ms": 5.0,
            "stages": [{"name": "pipeline.failed", "elapsed_ms": 5.0, "data": {"source_path": "bad.pdf"}}],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\nnot-json\n", encoding="utf-8")


def test_trace_service_filters_and_summarizes_ingestion_traces(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "traces.jsonl"
    write_trace_log(log_file)
    service = TraceService(log_file)

    summaries = service.ingestion_summaries()

    assert [summary["trace_id"] for summary in summaries] == ["ingest-new", "ingest-old", "ingest-failed"]
    assert summaries[0]["status"] == "success"
    assert summaries[0]["source_path"] == "docs/new.pdf"
    assert summaries[0]["collection"] == "docs"
    assert summaries[-1]["status"] == "failed"


def test_trace_service_stage_rows_include_details(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "traces.jsonl"
    write_trace_log(log_file)
    service = TraceService(log_file)

    rows = service.stage_rows(service.get_trace("ingest-new"))

    assert [row["stage"] for row in rows] == [
        "ingestion.dashboard.start",
        "batch_processor.process",
        "image_storage.index",
        "ingestion.dashboard.completed",
    ]
    assert rows[1]["provider"] == "local"
    assert rows[1]["details"]["chunk_count"] == 3


def test_ingestion_traces_model_selects_latest_and_extracts_stats(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "traces.jsonl"
    write_trace_log(log_file)

    model = ingestion_traces_model(trace_log_file=log_file)

    assert model["selected_summary"]["trace_id"] == "ingest-new"
    assert model["stats"] == {"chunk_count": 3, "image_count": 2, "skipped": False, "failed": False}
    assert model["waterfall_rows"][1] == {"stage": "batch_processor.process", "elapsed_ms": 8.0}


def test_ingestion_traces_model_can_select_trace_id(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "traces.jsonl"
    write_trace_log(log_file)

    model = ingestion_traces_model(trace_log_file=log_file, selected_trace_id="ingest-failed")

    assert model["selected_summary"]["status"] == "failed"
    assert model["stats"]["failed"] is True


def test_ingestion_traces_render_without_streamlit_returns_model(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "traces.jsonl"
    write_trace_log(log_file)

    rendered = render(trace_log_file=log_file)

    assert rendered["summaries"][0]["trace_id"] == "ingest-new"
