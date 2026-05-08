"""Composite evaluator for running multiple evaluation backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from libs.evaluator.base_evaluator import BaseEvaluator, EvaluationCase, EvaluationResult


class CompositeEvaluatorError(ValueError):
    """Raised when a composite evaluator is misconfigured."""


@dataclass(frozen=True)
class NamedEvaluator:
    """Evaluator paired with a stable result namespace."""

    name: str
    evaluator: BaseEvaluator


class CompositeEvaluator(BaseEvaluator):
    """Run multiple evaluators and merge their metrics/details."""

    def __init__(self, evaluators: Iterable[BaseEvaluator | NamedEvaluator]) -> None:
        self.evaluators = _normalize_evaluators(evaluators)
        if not self.evaluators:
            raise CompositeEvaluatorError("CompositeEvaluator requires at least one evaluator")

    def evaluate(self, case: EvaluationCase, trace: Any | None = None) -> EvaluationResult:
        """Evaluate a case with every child evaluator and merge results."""
        if not isinstance(case, EvaluationCase):
            raise CompositeEvaluatorError("case must be an EvaluationCase")
        metrics: dict[str, float] = {}
        details: dict[str, Any] = {"evaluators": []}
        for item in self.evaluators:
            result = item.evaluator.evaluate(case, trace=trace)
            if not isinstance(result, EvaluationResult):
                raise CompositeEvaluatorError(f"evaluator '{item.name}' returned {type(result).__name__}")
            _merge_metrics(metrics, item.name, result.metrics)
            details["evaluators"].append(
                {
                    "name": item.name,
                    "metrics": dict(result.metrics),
                    "details": dict(result.details),
                }
            )
        _record_trace(
            trace,
            "composite_evaluator.evaluate",
            {"evaluator_count": len(self.evaluators), "metric_count": len(metrics)},
        )
        return EvaluationResult(metrics=metrics, details=details)


def _normalize_evaluators(evaluators: Iterable[BaseEvaluator | NamedEvaluator]) -> list[NamedEvaluator]:
    normalized = []
    for index, item in enumerate(evaluators, start=1):
        if isinstance(item, NamedEvaluator):
            name = _normalize_name(item.name, index)
            evaluator = item.evaluator
        else:
            name = _normalize_name(item.__class__.__name__, index)
            evaluator = item
        if not isinstance(evaluator, BaseEvaluator):
            raise CompositeEvaluatorError("evaluators must contain BaseEvaluator instances")
        normalized.append(NamedEvaluator(name=name, evaluator=evaluator))
    return normalized


def _merge_metrics(target: dict[str, float], evaluator_name: str, metrics: dict[str, float]) -> None:
    if not isinstance(metrics, dict):
        raise CompositeEvaluatorError(f"metrics from '{evaluator_name}' must be a dict")
    for name, value in metrics.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise CompositeEvaluatorError(f"metric '{name}' from '{evaluator_name}' must be numeric") from exc
        key = name if name not in target else f"{evaluator_name}.{name}"
        target[key] = numeric


def _normalize_name(name: str, index: int) -> str:
    value = name.strip().lower().replace("evaluator", "") if isinstance(name, str) else ""
    return value or f"evaluator_{index}"


def _record_trace(trace: Any | None, name: str, data: dict[str, Any]) -> None:
    if hasattr(trace, "record_stage"):
        trace.record_stage(name, data)


__all__ = ["CompositeEvaluator", "CompositeEvaluatorError", "NamedEvaluator"]
