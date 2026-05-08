"""Tests for Ragas evaluator integration."""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import EvaluationSettings
from core.trace import TraceContext
from libs.evaluator import EvaluationCase, EvaluatorFactory, RagasEvaluator, RagasEvaluatorError


@pytest.fixture(autouse=True)
def reset_factory_registry() -> None:
    EvaluatorFactory.reset_defaults()


def case() -> EvaluationCase:
    return EvaluationCase(
        query="How do I configure Azure?",
        retrieved_ids=["chunk-a", "chunk-b"],
        golden_ids=["chunk-a"],
        metadata={
            "answer": "Set the Azure deployment and API key in settings.",
            "ground_truth": "Configure Azure deployment and credentials in settings.",
            "contexts": ["Azure settings include deployment, endpoint, and API key."],
        },
    )


def test_ragas_evaluator_uses_runner_and_normalizes_metrics() -> None:
    calls: list[tuple[dict[str, Any], tuple[str, ...]]] = []

    def runner(payload: dict[str, Any], metrics: tuple[str, ...]) -> dict[str, float]:
        calls.append((payload, metrics))
        return {"faithfulness": 0.8, "answer_relevancy": 0.7, "context_precision": 0.6}

    result = RagasEvaluator(runner=runner).evaluate(case())

    assert result.metrics == {"faithfulness": 0.8, "answer_relevancy": 0.7, "context_precision": 0.6}
    assert result.details["context_count"] == 1
    assert result.details["metric_names"] == ["faithfulness", "answer_relevancy", "context_precision"]
    assert calls[0][0]["question"] == "How do I configure Azure?"
    assert calls[0][0]["contexts"] == ["Azure settings include deployment, endpoint, and API key."]


def test_ragas_evaluator_records_trace_stage() -> None:
    trace = TraceContext(trace_type="query")

    RagasEvaluator(runner=lambda payload, metrics: {"faithfulness": 1, "answer_relevancy": 0.5}).evaluate(
        case(),
        trace=trace,
    )

    assert trace.stages[-1] == {
        "name": "ragas_evaluator.evaluate",
        "data": {"metrics": {"faithfulness": 1.0, "answer_relevancy": 0.5}, "context_count": 1},
    }


def test_ragas_evaluator_requires_answer_and_contexts() -> None:
    evaluator = RagasEvaluator(runner=lambda payload, metrics: {"faithfulness": 1.0})

    with pytest.raises(RagasEvaluatorError, match="answer"):
        evaluator.evaluate(EvaluationCase(query="q", retrieved_ids=[], golden_ids=[], metadata={"contexts": ["ctx"]}))

    with pytest.raises(RagasEvaluatorError, match="contexts"):
        evaluator.evaluate(EvaluationCase(query="q", retrieved_ids=[], golden_ids=[], metadata={"answer": "a"}))


def test_ragas_evaluator_rejects_non_numeric_metrics() -> None:
    evaluator = RagasEvaluator(runner=lambda payload, metrics: {"faithfulness": "high"})

    with pytest.raises(RagasEvaluatorError, match="numeric"):
        evaluator.evaluate(case())


def test_ragas_evaluator_missing_dependency_has_readable_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "datasets":
            raise ImportError("missing datasets")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ImportError, match="pip install ragas datasets"):
        RagasEvaluator().evaluate(case())


def test_evaluator_factory_registers_ragas_backend() -> None:
    evaluator = EvaluatorFactory.create(EvaluationSettings(backends=["ragas"], golden_test_set="./golden.json"))

    assert isinstance(evaluator, RagasEvaluator)
