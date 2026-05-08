"""Ragas-backed evaluator integration."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from libs.evaluator.base_evaluator import BaseEvaluator, EvaluationCase, EvaluationResult


RAGAS_METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision")
RagasRunner = Callable[[dict[str, Any], tuple[str, ...]], Mapping[str, Any] | Any]


class RagasEvaluatorError(ValueError):
    """Raised when a Ragas evaluation case is incomplete or malformed."""


class RagasEvaluator(BaseEvaluator):
    """Evaluate RAG answers with Ragas metrics.

    The real Ragas package is imported only when no runner is injected. This keeps
    normal test and offline workflows dependency-light while still providing a
    production adapter for the Ragas framework.
    """

    def __init__(self, runner: RagasRunner | None = None, metric_names: tuple[str, ...] = RAGAS_METRIC_NAMES) -> None:
        self.runner = runner or _run_ragas
        self.metric_names = tuple(metric_names)

    def evaluate(self, case: EvaluationCase, trace: Any | None = None) -> EvaluationResult:
        """Evaluate a single RAG case and return normalized metric values."""
        payload = _case_payload(case)
        raw_result = self.runner(payload, self.metric_names)
        metrics = _normalize_metrics(raw_result, self.metric_names)
        details = {
            "query": case.query,
            "metric_names": list(self.metric_names),
            "context_count": len(payload["contexts"]),
            "raw_result": _json_safe(raw_result),
        }
        _record_trace(
            trace,
            "ragas_evaluator.evaluate",
            {"metrics": metrics, "context_count": details["context_count"]},
        )
        return EvaluationResult(metrics=metrics, details=details)


def _case_payload(case: EvaluationCase) -> dict[str, Any]:
    if not isinstance(case, EvaluationCase):
        raise RagasEvaluatorError("case must be an EvaluationCase")
    question = _non_empty("query", case.query)
    answer = _metadata_text(case.metadata, "answer", "generated_answer", "response")
    ground_truth = _metadata_text(case.metadata, "ground_truth", "reference", "expected_answer", required=False)
    contexts = _metadata_list(case.metadata, "contexts", "retrieved_contexts", "context")
    if not contexts:
        raise RagasEvaluatorError("Ragas evaluation requires contexts in case.metadata['contexts']")
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
        "retrieved_ids": list(case.retrieved_ids),
        "golden_ids": list(case.golden_ids),
    }


def _run_ragas(payload: dict[str, Any], metric_names: tuple[str, ...]) -> Mapping[str, Any]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError as exc:
        raise ImportError(
            "RagasEvaluator requires optional dependencies. "
            "Install them with `pip install ragas datasets` or inject a test runner."
        ) from exc

    metric_registry = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
    }
    selected_metrics = [metric_registry[name] for name in metric_names if name in metric_registry]
    dataset = Dataset.from_dict(
        {
            "question": [payload["question"]],
            "answer": [payload["answer"]],
            "contexts": [payload["contexts"]],
            "ground_truth": [payload["ground_truth"] or ""],
        }
    )
    result = evaluate(dataset, metrics=selected_metrics)
    if hasattr(result, "to_pandas"):
        row = result.to_pandas().iloc[0].to_dict()
        return {name: row.get(name) for name in metric_names}
    if isinstance(result, Mapping):
        return result
    return {name: getattr(result, name, None) for name in metric_names}


def _normalize_metrics(raw_result: Mapping[str, Any] | Any, metric_names: tuple[str, ...]) -> dict[str, float]:
    if hasattr(raw_result, "to_dict") and not isinstance(raw_result, Mapping):
        raw_result = raw_result.to_dict()
    if not isinstance(raw_result, Mapping):
        raise RagasEvaluatorError("Ragas runner must return a mapping of metric names to scores")
    metrics: dict[str, float] = {}
    for name in metric_names:
        if name not in raw_result:
            continue
        value = raw_result[name]
        if isinstance(value, list):
            value = value[0] if value else None
        if value is None:
            continue
        try:
            metrics[name] = float(value)
        except (TypeError, ValueError) as exc:
            raise RagasEvaluatorError(f"Ragas metric '{name}' must be numeric") from exc
    if not metrics:
        raise RagasEvaluatorError("Ragas runner returned no recognized metrics")
    return metrics


def _metadata_text(metadata: dict[str, Any], *keys: str, required: bool = True) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if required:
        joined = ", ".join(keys)
        raise RagasEvaluatorError(f"Ragas evaluation requires one of metadata keys: {joined}")
    return None


def _metadata_list(metadata: dict[str, Any], *keys: str) -> list[str]:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        if isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
            if items:
                return items
    return []


def _non_empty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RagasEvaluatorError(f"{name} must be a non-empty string")
    return value.strip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _record_trace(trace: Any | None, name: str, data: dict[str, Any]) -> None:
    if hasattr(trace, "record_stage"):
        trace.record_stage(name, data)


__all__ = ["RAGAS_METRIC_NAMES", "RagasEvaluator", "RagasEvaluatorError"]
