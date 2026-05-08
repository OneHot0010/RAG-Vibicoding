"""Trace data service for dashboard pages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceSummary:
    """Compact trace row for dashboard history lists."""

    trace_id: str
    trace_type: str
    started_at: str | None
    finished_at: str | None
    total_elapsed_ms: float
    status: str
    source_path: str | None = None
    collection: str | None = None
    query: str | None = None
    stage_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize summary row."""
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": self.total_elapsed_ms,
            "status": self.status,
            "source_path": self.source_path,
            "collection": self.collection,
            "query": self.query,
            "stage_count": self.stage_count,
        }


class TraceService:
    """Read and shape JSON Lines trace records."""

    def __init__(self, trace_log_file: str | Path = "logs/traces.jsonl") -> None:
        self.trace_log_file = Path(trace_log_file)

    def list_traces(self, trace_type: str | None = None) -> list[dict[str, Any]]:
        """Return trace payloads, optionally filtered by type."""
        traces = self._read_traces()
        if trace_type is not None:
            traces = [trace for trace in traces if trace.get("trace_type") == trace_type]
        return sorted(traces, key=lambda trace: str(trace.get("started_at") or ""), reverse=True)

    def list_summaries(self, trace_type: str | None = None) -> list[dict[str, Any]]:
        """Return compact trace summaries."""
        return [self.summarize_trace(trace).to_dict() for trace in self.list_traces(trace_type=trace_type)]

    def get_trace(self, trace_id: str) -> dict[str, Any]:
        """Return one trace by id."""
        for trace in self._read_traces():
            if trace.get("trace_id") == trace_id:
                return trace
        raise ValueError(f"trace not found: {trace_id}")

    def summarize_trace(self, trace: dict[str, Any]) -> TraceSummary:
        """Build a compact summary for one trace."""
        source_path, collection = _trace_context_fields(trace)
        query = _trace_query(trace)
        return TraceSummary(
            trace_id=str(trace.get("trace_id") or ""),
            trace_type=str(trace.get("trace_type") or ""),
            started_at=_optional_str(trace.get("started_at")),
            finished_at=_optional_str(trace.get("finished_at")),
            total_elapsed_ms=float(trace.get("total_elapsed_ms") or 0.0),
            status=_trace_status(trace),
            source_path=source_path,
            collection=collection,
            query=query,
            stage_count=len(_stages(trace)),
        )

    def stage_rows(self, trace: dict[str, Any]) -> list[dict[str, Any]]:
        """Return stage rows for tables and charts."""
        rows = []
        for index, stage in enumerate(_stages(trace), start=1):
            data = stage.get("data") if isinstance(stage.get("data"), dict) else {}
            rows.append(
                {
                    "index": index,
                    "stage": str(stage.get("name") or ""),
                    "elapsed_ms": round(float(stage.get("elapsed_ms") or data.get("elapsed_ms") or 0.0), 3),
                    "started_at": stage.get("started_at"),
                    "method": data.get("method"),
                    "provider": data.get("provider"),
                    "details": data,
                }
            )
        return rows

    def ingestion_summaries(self) -> list[dict[str, Any]]:
        """Return ingestion trace summaries."""
        return self.list_summaries(trace_type="ingestion")

    def query_summaries(self) -> list[dict[str, Any]]:
        """Return query trace summaries."""
        return self.list_summaries(trace_type="query")

    def _read_traces(self) -> list[dict[str, Any]]:
        if not self.trace_log_file.is_file():
            return []
        traces = []
        for line in self.trace_log_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                traces.append(decoded)
        return traces


def _trace_context_fields(trace: dict[str, Any]) -> tuple[str | None, str | None]:
    source_path = _optional_str(trace.get("source_path"))
    collection = _optional_str(trace.get("collection"))
    for stage in _stages(trace):
        data = stage.get("data") if isinstance(stage.get("data"), dict) else {}
        source_path = source_path or _optional_str(data.get("source_path"))
        collection = collection or _optional_str(data.get("collection"))
        filters = data.get("filters")
        if collection is None and isinstance(filters, dict):
            collection = _optional_str(filters.get("collection"))
        if source_path and collection:
            break
    return source_path, collection


def _trace_status(trace: dict[str, Any]) -> str:
    stage_names = [str(stage.get("name") or "") for stage in _stages(trace)]
    if any("failed" in name for name in stage_names):
        return "failed"
    if any("skipped" in name for name in stage_names):
        return "skipped"
    if trace.get("finished_at") or any("completed" in name for name in stage_names):
        return "success"
    return "running"


def _trace_query(trace: dict[str, Any]) -> str | None:
    query = _optional_str(trace.get("query") or trace.get("user_query"))
    if query:
        return query
    for stage in _stages(trace):
        data = stage.get("data") if isinstance(stage.get("data"), dict) else {}
        query = _optional_str(data.get("query") or data.get("user_query") or data.get("original_query"))
        if query:
            return query
    return None


def _stages(trace: dict[str, Any]) -> list[dict[str, Any]]:
    stages = trace.get("stages")
    return [stage for stage in stages if isinstance(stage, dict)] if isinstance(stages, list) else []


def _optional_str(value: Any) -> str | None:
    return str(value) if value not in {None, ""} else None


__all__ = ["TraceService", "TraceSummary"]
