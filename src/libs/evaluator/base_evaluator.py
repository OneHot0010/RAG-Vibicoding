"""Base abstractions for evaluation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationCase:
    """A single retrieval evaluation case."""

    query: str
    retrieved_ids: list[str]
    golden_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics returned by an evaluator."""

    metrics: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)


class BaseEvaluator(ABC):
    """Common interface for evaluation backends."""

    @abstractmethod
    def evaluate(self, case: EvaluationCase, trace: Any | None = None) -> EvaluationResult:
        """Evaluate a single retrieval case."""
