"""Tests for JSON Lines trace logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from core.trace import TraceCollector, TraceContext
from observability.logger import JSONFormatter, get_trace_logger, write_trace


def trace_payload() -> dict:
    trace = TraceContext(trace_type="query")
    trace.record_stage("query_processor", {"keyword_count": 2})
    trace.finish()
    return trace.to_dict()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_json_formatter_outputs_dict_message_as_top_level_json() -> None:
    record = logging.LogRecord(
        name="trace",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg={"trace_id": "abc", "trace_type": "query"},
        args=(),
        exc_info=None,
    )

    decoded = json.loads(JSONFormatter().format(record))

    assert decoded == {"trace_id": "abc", "trace_type": "query"}


def test_json_formatter_outputs_plain_messages_as_structured_json() -> None:
    record = logging.LogRecord(
        name="trace",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )

    decoded = json.loads(JSONFormatter().format(record))

    assert decoded == {"level": "WARNING", "logger": "trace", "message": "hello"}


def test_write_trace_appends_one_json_line_with_trace_type(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "traces.jsonl"
    payload = trace_payload()

    write_trace(payload, log_file=log_file)

    rows = read_jsonl(log_file)
    assert len(rows) == 1
    assert rows[0]["trace_id"] == payload["trace_id"]
    assert rows[0]["trace_type"] == "query"
    assert rows[0]["stages"][0]["name"] == "query_processor"


def test_write_trace_appends_multiple_lines(tmp_path: Path) -> None:
    log_file = tmp_path / "traces.jsonl"
    first = trace_payload()
    second = {**trace_payload(), "trace_id": "second", "trace_type": "ingestion"}

    write_trace(first, log_file=log_file)
    write_trace(second, log_file=log_file)

    rows = read_jsonl(log_file)
    assert [row["trace_id"] for row in rows] == [first["trace_id"], "second"]
    assert rows[1]["trace_type"] == "ingestion"


def test_get_trace_logger_does_not_duplicate_handlers_for_same_file(tmp_path: Path) -> None:
    log_file = tmp_path / "traces.jsonl"
    logger = get_trace_logger(log_file=log_file, name=f"trace.test.{id(log_file)}")
    same_logger = get_trace_logger(log_file=log_file, name=f"trace.test.{id(log_file)}")

    same_logger.info({"trace_id": "one", "trace_type": "query"})

    assert logger is same_logger
    assert len(read_jsonl(log_file)) == 1


def test_trace_collector_can_persist_through_write_trace_sink(tmp_path: Path) -> None:
    log_file = tmp_path / "traces.jsonl"
    collector = TraceCollector(sink=lambda payload: write_trace(payload, log_file=log_file))
    trace = TraceContext(trace_type="ingestion")
    trace.record_stage("loader.load", {"source_path": "docs/a.pdf"})

    collector.collect(trace)

    rows = read_jsonl(log_file)
    assert rows == collector.traces
    assert rows[0]["trace_type"] == "ingestion"
    assert rows[0]["finished_at"] is not None


def test_write_trace_rejects_non_dict_payload(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="trace_dict"):
        write_trace(["bad"], log_file=tmp_path / "traces.jsonl")  # type: ignore[arg-type]
