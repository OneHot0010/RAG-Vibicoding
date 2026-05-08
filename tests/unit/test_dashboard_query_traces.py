"""Tests for dashboard query trace page model."""

from __future__ import annotations

import json
from pathlib import Path

from observability.dashboard.pages.query_traces import query_traces_model, render
from observability.dashboard.services.trace_service import TraceService


def write_trace_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "trace_id": "ingest-1",
            "trace_type": "ingestion",
            "started_at": "2026-05-08T09:00:00",
            "finished_at": "2026-05-08T09:00:01",
            "total_elapsed_ms": 1.0,
            "stages": [],
        },
        {
            "trace_id": "query-old",
            "trace_type": "query",
            "started_at": "2026-05-08T10:00:00",
            "finished_at": "2026-05-08T10:00:01",
            "total_elapsed_ms": 10.0,
            "stages": [
                {
                    "name": "query.start",
                    "elapsed_ms": 1.0,
                    "data": {"query": "old query", "filters": {"collection": "docs"}},
                },
                {"name": "query.completed", "elapsed_ms": 1.0, "data": {"result_count": 0, "rerank_enabled": False}},
            ],
        },
        {
            "trace_id": "query-new",
            "trace_type": "query",
            "started_at": "2026-05-08T11:00:00",
            "finished_at": "2026-05-08T11:00:02",
            "total_elapsed_ms": 25.0,
            "stages": [
                {
                    "name": "query.start",
                    "elapsed_ms": 1.0,
                    "data": {"query": "Azure config", "filters": {"collection": "docs"}},
                },
                {
                    "name": "dense_retriever.retrieve",
                    "elapsed_ms": 5.0,
                    "data": {
                        "result_count": 2,
                        "provider": "fake",
                        "dense_results": [
                            {
                                "chunk_id": "chunk-a",
                                "score": 0.9,
                                "text": "dense alpha",
                                "metadata": {"source_path": "docs/a.md"},
                            },
                            {
                                "chunk_id": "chunk-b",
                                "score": 0.7,
                                "text": "dense beta",
                                "metadata": {"source_path": "docs/b.md"},
                            },
                        ],
                    },
                },
                {
                    "name": "sparse_retriever.retrieve",
                    "elapsed_ms": 4.0,
                    "data": {
                        "result_count": 1,
                        "sparse_results": [
                            {
                                "chunk_id": "chunk-b",
                                "score": 3.2,
                                "text": "sparse beta",
                                "metadata": {"source_path": "docs/b.md"},
                            }
                        ],
                    },
                },
                {
                    "name": "fusion.rrf",
                    "elapsed_ms": 2.0,
                    "data": {
                        "result_count": 2,
                        "fusion_results": [
                            {"chunk_id": "chunk-a", "score": 0.03, "text": "fused alpha", "metadata": {"source_path": "docs/a.md"}},
                            {"chunk_id": "chunk-b", "score": 0.02, "text": "fused beta", "metadata": {"source_path": "docs/b.md"}},
                        ],
                    },
                },
                {
                    "name": "core_reranker.rerank",
                    "elapsed_ms": 6.0,
                    "data": {
                        "result_count": 2,
                        "rerank_results": [
                            {"chunk_id": "chunk-b", "score": 9.0, "text": "rerank beta", "metadata": {"source_path": "docs/b.md"}},
                            {"chunk_id": "chunk-a", "score": 8.0, "text": "rerank alpha", "metadata": {"source_path": "docs/a.md"}},
                        ],
                    },
                },
                {
                    "name": "query.completed",
                    "elapsed_ms": 1.0,
                    "data": {"fused_count": 2, "result_count": 2, "rerank_enabled": True},
                },
            ],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_trace_service_query_summaries_include_query_text(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "traces.jsonl"
    write_trace_log(log_file)

    summaries = TraceService(log_file).query_summaries()

    assert [summary["trace_id"] for summary in summaries] == ["query-new", "query-old"]
    assert summaries[0]["query"] == "Azure config"
    assert summaries[0]["collection"] == "docs"


def test_query_traces_model_extracts_route_and_rerank_rows(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "traces.jsonl"
    write_trace_log(log_file)

    model = query_traces_model(trace_log_file=log_file)

    assert model["selected_summary"]["trace_id"] == "query-new"
    assert model["stats"]["dense_count"] == 2
    assert model["stats"]["sparse_count"] == 1
    assert model["stats"]["result_count"] == 2
    assert model["stats"]["rerank_enabled"] is True
    assert [row["chunk_id"] for row in model["dense_rows"]] == ["chunk-a", "chunk-b"]
    assert [row["chunk_id"] for row in model["sparse_rows"]] == ["chunk-b"]
    assert [row["chunk_id"] for row in model["fusion_rows"]] == ["chunk-a", "chunk-b"]
    assert model["rerank_rows"][0]["chunk_id"] == "chunk-b"
    assert model["rerank_rows"][0]["previous_rank"] == 2
    assert model["rerank_rows"][0]["rank_delta"] == 1


def test_query_traces_model_filters_search_and_selects_trace(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "traces.jsonl"
    write_trace_log(log_file)

    model = query_traces_model(trace_log_file=log_file, search="old")

    assert [summary["trace_id"] for summary in model["summaries"]] == ["query-old"]
    assert model["selected_summary"]["query"] == "old query"


def test_query_traces_render_without_streamlit_returns_model(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "traces.jsonl"
    write_trace_log(log_file)

    rendered = render(trace_log_file=log_file)

    assert rendered["waterfall_rows"][1] == {"stage": "dense_retriever.retrieve", "elapsed_ms": 5.0}
