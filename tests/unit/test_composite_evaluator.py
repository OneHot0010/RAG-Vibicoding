"""Tests for composite evaluator orchestration."""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import EvaluationSettings
from core.trace import TraceContext
from libs.evaluator import BaseEvaluator, EvaluationCase, EvaluationResult, EvaluatorFactory
from observability.evaluation import CompositeEvaluator, CompositeEvaluatorError, NamedEvaluator


class StaticEvaluator(BaseEvaluator):
    def __init__(self, metrics: dict[str, float], label: str = "static") -> None:
        self.metrics = metrics
        self.label = label
        self.calls: list[EvaluationCase] = []

    def evaluate(self, case: EvaluationCase, trace: Any | None = None) -> EvaluationResult:
        self.calls.append(case)
        return EvaluationResult(metrics=self.metrics, details={"label": self.label})


class BadEvaluator(BaseEvaluator):
    def evaluate(self, case: EvaluationCase, trace: Any | None = None) -> EvaluationResult:
        return "bad"  # type: ignore[return-value]


@pytest.fixture(autouse=True)
def reset_factory_registry() -> None:
    EvaluatorFactory.reset_defaults()


def case() -> EvaluationCase:
    return EvaluationCase(query="config", retrieved_ids=["doc-a"], golden_ids=["doc-a"])


def test_composite_evaluator_merges_metrics_and_details() -> None:
    first = StaticEvaluator({"hit_rate": 1.0}, label="retrieval")
    second = StaticEvaluator({"faithfulness": 0.8}, label="ragas")

    result = CompositeEvaluator([NamedEvaluator("custom", first), NamedEvaluator("ragas", second)]).evaluate(case())

    assert result.metrics == {"hit_rate": 1.0, "faithfulness": 0.8}
    assert result.details["evaluators"] == [
        {"name": "custom", "metrics": {"hit_rate": 1.0}, "details": {"label": "retrieval"}},
        {"name": "ragas", "metrics": {"faithfulness": 0.8}, "details": {"label": "ragas"}},
    ]
    assert first.calls == [case()]
    assert second.calls == [case()]


def test_composite_evaluator_prefixes_duplicate_metric_names() -> None:
    result = CompositeEvaluator(
        [
            NamedEvaluator("first", StaticEvaluator({"score": 0.1})),
            NamedEvaluator("second", StaticEvaluator({"score": 0.2})),
        ]
    ).evaluate(case())

    assert result.metrics == {"score": 0.1, "second.score": 0.2}


def test_composite_evaluator_records_trace_stage() -> None:
    trace = TraceContext(trace_type="query")

    CompositeEvaluator([NamedEvaluator("custom", StaticEvaluator({"hit_rate": 1.0}))]).evaluate(case(), trace=trace)

    assert trace.stages[-1] == {
        "name": "composite_evaluator.evaluate",
        "data": {"evaluator_count": 1, "metric_count": 1},
    }


def test_composite_evaluator_validates_inputs_and_results() -> None:
    with pytest.raises(CompositeEvaluatorError, match="at least one"):
        CompositeEvaluator([])
    with pytest.raises(CompositeEvaluatorError, match="BaseEvaluator"):
        CompositeEvaluator(["bad"])  # type: ignore[list-item]
    with pytest.raises(CompositeEvaluatorError, match="returned str"):
        CompositeEvaluator([BadEvaluator()]).evaluate(case())
    with pytest.raises(CompositeEvaluatorError, match="numeric"):
        CompositeEvaluator([StaticEvaluator({"bad": "high"})]).evaluate(case())  # type: ignore[arg-type]


def test_factory_returns_composite_for_multiple_configured_backends() -> None:
    EvaluatorFactory.register("fake", lambda settings: StaticEvaluator({"fake_score": 0.5}))

    evaluator = EvaluatorFactory.create(EvaluationSettings(backends=["custom", "fake"], golden_test_set="./golden.json"))

    assert isinstance(evaluator, CompositeEvaluator)
    result = evaluator.evaluate(EvaluationCase(query="q", retrieved_ids=["a"], golden_ids=["a"]))
    assert result.metrics["hit_rate"] == 1.0
    assert result.metrics["fake_score"] == 0.5


def test_factory_explicit_backend_still_returns_single_evaluator() -> None:
    EvaluatorFactory.register("fake", lambda settings: StaticEvaluator({"fake_score": 0.5}))

    evaluator = EvaluatorFactory.create(
        EvaluationSettings(backends=["custom", "fake"], golden_test_set="./golden.json"),
        backend="fake",
    )

    assert isinstance(evaluator, StaticEvaluator)
