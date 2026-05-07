"""Minimal trace context used by ingestion components."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceContext:
    """Small stage recorder; later phases can enrich this without changing callers."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    stages: list[dict[str, Any]] = field(default_factory=list)

    def record_stage(self, name: str, data: dict[str, Any] | None = None) -> None:
        """Append a stage event for observability."""
        self.stages.append({"name": name, "data": data or {}})
