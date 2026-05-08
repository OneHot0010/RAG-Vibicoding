"""In-memory trace collection hook."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.trace.trace_context import TraceContext


class TraceCollector:
    """Collect completed traces and optionally forward serialized payloads."""

    def __init__(self, sink: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.sink = sink
        self.traces: list[dict[str, Any]] = []

    def collect(self, trace: TraceContext) -> None:
        """Finish, serialize, store, and optionally forward one trace."""
        if not isinstance(trace, TraceContext):
            raise ValueError("trace must be a TraceContext")
        trace.finish()
        payload = trace.to_dict()
        self.traces.append(payload)
        if self.sink is not None:
            self.sink(payload)
