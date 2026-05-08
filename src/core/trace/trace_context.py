"""Request trace context shared by ingestion and query flows."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TraceContext:
    """Record request-level stages and serialize trace timing data."""

    trace_type: str = "query"
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    stages: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: _now_iso())
    finished_at: str | None = None
    total_elapsed_ms: float | None = None
    _stage_timings: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _started_monotonic: float = field(default_factory=time.perf_counter, repr=False)
    _last_stage_monotonic: float = field(default_factory=time.perf_counter, repr=False)

    def __post_init__(self) -> None:
        self.trace_type = _validate_trace_type(self.trace_type)

    def record_stage(self, name: str, data: dict[str, Any] | None = None) -> None:
        """Append a stage event for observability."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("stage name must be a non-empty string")
        if data is not None and not isinstance(data, dict):
            raise ValueError("stage data must be a dict when provided")

        now = time.perf_counter()
        stage_elapsed_ms = (now - self._last_stage_monotonic) * 1000
        self._last_stage_monotonic = now
        self.stages.append({"name": name.strip(), "data": dict(data or {})})
        self._stage_timings.append({"started_at": _now_iso(), "elapsed_ms": stage_elapsed_ms})

    def finish(self) -> None:
        """Mark the trace as completed and compute total elapsed time."""
        if self.finished_at is not None:
            return
        self.finished_at = _now_iso()
        self.total_elapsed_ms = (time.perf_counter() - self._started_monotonic) * 1000

    def elapsed_ms(self, stage_name: str | None = None) -> float:
        """Return elapsed milliseconds for the whole trace or the latest matching stage."""
        if stage_name is None:
            if self.total_elapsed_ms is not None:
                return self.total_elapsed_ms
            return (time.perf_counter() - self._started_monotonic) * 1000
        matches = [stage for stage in self.stages if stage["name"] == stage_name]
        if not matches:
            return 0.0
        index = max(index for index, stage in enumerate(self.stages) if stage["name"] == stage_name)
        return float(self._stage_timings[index]["elapsed_ms"])

    def to_dict(self) -> dict[str, Any]:
        """Serialize this trace into JSON-friendly primitives."""
        total_elapsed_ms = self.total_elapsed_ms if self.total_elapsed_ms is not None else self.elapsed_ms()
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": total_elapsed_ms,
            "stages": [
                _serialize_stage(stage, self._stage_timings[index])
                for index, stage in enumerate(self.stages)
            ],
        }


def _validate_trace_type(trace_type: str) -> str:
    if trace_type not in {"query", "ingestion"}:
        raise ValueError("trace_type must be 'query' or 'ingestion'")
    return trace_type


def _serialize_stage(stage: dict[str, Any], timing: dict[str, Any]) -> dict[str, Any]:
    elapsed_ms = float(timing.get("elapsed_ms") or 0.0)
    data = dict(stage.get("data") or {})
    data.setdefault("elapsed_ms", elapsed_ms)
    return {
        "name": stage["name"],
        "data": data,
        "started_at": timing.get("started_at"),
        "elapsed_ms": elapsed_ms,
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
