"""Tests for TraceContext timing and serialization."""

from __future__ import annotations

import json
import time

import pytest

from core.trace import TraceCollector, TraceContext


def test_trace_context_defaults_to_query_trace() -> None:
    trace = TraceContext()

    assert trace.trace_type == "query"
    assert trace.trace_id
    assert trace.started_at
    assert trace.finished_at is None


def test_trace_context_accepts_ingestion_trace_type() -> None:
    trace = TraceContext(trace_type="ingestion")

    assert trace.trace_type == "ingestion"


def test_trace_context_rejects_invalid_trace_type() -> None:
    with pytest.raises(ValueError, match="trace_type"):
        TraceContext(trace_type="other")


def test_record_stage_adds_timing_and_preserves_data_shape() -> None:
    trace = TraceContext()
    time.sleep(0.001)

    trace.record_stage("dense_retrieval", {"result_count": 3})

    stage = trace.stages[0]
    assert stage == {"name": "dense_retrieval", "data": {"result_count": 3}}
    assert trace.elapsed_ms("dense_retrieval") >= 0


def test_record_stage_validates_name_and_data() -> None:
    trace = TraceContext()

    with pytest.raises(ValueError, match="stage name"):
        trace.record_stage("")
    with pytest.raises(ValueError, match="stage data"):
        trace.record_stage("bad", [])  # type: ignore[arg-type]


def test_finish_is_idempotent_and_to_dict_is_json_serializable() -> None:
    trace = TraceContext(trace_type="ingestion")
    trace.record_stage("load", {"method": "pdf"})
    trace.finish()
    first_finished_at = trace.finished_at
    first_elapsed = trace.total_elapsed_ms
    trace.finish()

    payload = trace.to_dict()

    assert trace.finished_at == first_finished_at
    assert trace.total_elapsed_ms == first_elapsed
    assert payload["trace_id"] == trace.trace_id
    assert payload["trace_type"] == "ingestion"
    assert payload["started_at"] == trace.started_at
    assert payload["finished_at"] == first_finished_at
    assert payload["total_elapsed_ms"] == first_elapsed
    assert payload["stages"][0]["name"] == "load"
    assert payload["stages"][0]["data"]["method"] == "pdf"
    assert payload["stages"][0]["data"]["elapsed_ms"] >= 0
    assert payload["stages"][0]["elapsed_ms"] >= 0
    assert payload["stages"][0]["started_at"]
    json.dumps(payload)


def test_elapsed_ms_without_finish_returns_running_elapsed() -> None:
    trace = TraceContext()
    time.sleep(0.001)

    assert trace.elapsed_ms() > 0
    assert trace.elapsed_ms("missing") == 0.0


def test_trace_collector_finishes_and_forwards_payload() -> None:
    forwarded: list[dict] = []
    collector = TraceCollector(sink=forwarded.append)
    trace = TraceContext()
    trace.record_stage("query_processor", {"keyword_count": 2})

    collector.collect(trace)

    assert trace.finished_at is not None
    assert len(collector.traces) == 1
    assert forwarded == collector.traces
    assert forwarded[0]["stages"][0]["name"] == "query_processor"


def test_trace_collector_rejects_invalid_trace() -> None:
    collector = TraceCollector()

    with pytest.raises(ValueError, match="TraceContext"):
        collector.collect("bad")  # type: ignore[arg-type]
