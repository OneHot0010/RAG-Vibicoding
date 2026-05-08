"""Evaluation runner for golden test sets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from core.settings import Settings
from core.trace import TraceContext
from core.types import RetrievalResult
from libs.evaluator import BaseEvaluator, EvaluationCase, EvaluationResult


class EvalRunnerError(ValueError):
    """Raised when an evaluation run cannot be completed."""


@dataclass(frozen=True)
class GoldenTestCase:
    """One case from a golden test set."""

    query: str
    expected_chunk_ids: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    answer: str | None = None
    ground_truth: str | None = None
    contexts: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    top_k: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def golden_ids(self) -> list[str]:
        """Return all accepted ids for evaluator matching."""
        return _unique([*self.expected_chunk_ids, *self.expected_sources])

    def to_dict(self) -> dict[str, Any]:
        """Serialize this case."""
        return {
            "query": self.query,
            "expected_chunk_ids": list(self.expected_chunk_ids),
            "expected_sources": list(self.expected_sources),
            "answer": self.answer,
            "ground_truth": self.ground_truth,
            "contexts": list(self.contexts),
            "filters": dict(self.filters),
            "top_k": self.top_k,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EvalCaseResult:
    """Evaluation output for one golden case."""

    query: str
    metrics: dict[str, float]
    retrieved_ids: list[str]
    golden_ids: list[str]
    results: list[dict[str, Any]]
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize case result."""
        return {
            "query": self.query,
            "metrics": dict(self.metrics),
            "retrieved_ids": list(self.retrieved_ids),
            "golden_ids": list(self.golden_ids),
            "results": list(self.results),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class EvalReport:
    """Aggregate evaluation report."""

    metrics: dict[str, float]
    cases: list[EvalCaseResult]
    test_set_path: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize report."""
        return {
            "metrics": dict(self.metrics),
            "case_count": len(self.cases),
            "test_set_path": self.test_set_path,
            "cases": [case.to_dict() for case in self.cases],
        }


class EvalRunner:
    """Run retrieval and evaluator backends over a golden test set."""

    def __init__(
        self,
        settings: Settings,
        hybrid_search: Any,
        evaluator: BaseEvaluator,
        *,
        top_k: int | None = None,
    ) -> None:
        self.settings = settings
        self.hybrid_search = hybrid_search
        self.evaluator = evaluator
        self.top_k = top_k

    def run(self, test_set_path: str | Path, trace: TraceContext | None = None) -> EvalReport:
        """Run all cases in a golden test set and return an aggregate report."""
        path = Path(test_set_path)
        cases = load_golden_test_set(path)
        if not cases:
            raise EvalRunnerError(f"golden test set contains no test_cases: {path}")
        results = [self._run_case(case, trace=trace) for case in cases]
        report = EvalReport(metrics=_aggregate_metrics(results), cases=results, test_set_path=str(path))
        _record_trace(trace, "eval_runner.completed", {"case_count": len(results), "metrics": report.metrics})
        return report

    def _run_case(self, case: GoldenTestCase, trace: TraceContext | None = None) -> EvalCaseResult:
        top_k = case.top_k or self.top_k or self.settings.retrieval.top_k_final
        retrieved = self.hybrid_search.search(case.query, top_k=top_k, filters=case.filters, trace=trace)
        _validate_results(retrieved)
        retrieved_ids = _retrieved_ids(retrieved)
        contexts = case.contexts or [result.text for result in retrieved]
        metadata = {
            **case.metadata,
            "answer": case.answer or _default_answer(retrieved),
            "ground_truth": case.ground_truth,
            "contexts": contexts,
            "expected_sources": case.expected_sources,
            "filters": case.filters,
        }
        evaluation_case = EvaluationCase(
            query=case.query,
            retrieved_ids=retrieved_ids,
            golden_ids=case.golden_ids(),
            metadata=metadata,
        )
        evaluation = self.evaluator.evaluate(evaluation_case, trace=trace)
        if not isinstance(evaluation, EvaluationResult):
            raise EvalRunnerError(f"evaluator returned {type(evaluation).__name__}, expected EvaluationResult")
        return EvalCaseResult(
            query=case.query,
            metrics=dict(evaluation.metrics),
            retrieved_ids=retrieved_ids,
            golden_ids=evaluation_case.golden_ids,
            results=[result.to_dict() for result in retrieved],
            details=dict(evaluation.details),
        )


def load_golden_test_set(path: str | Path) -> list[GoldenTestCase]:
    """Load golden test cases from JSON."""
    test_set_path = Path(path)
    if not test_set_path.is_file():
        raise EvalRunnerError(f"golden test set not found: {test_set_path}")
    try:
        payload = json.loads(test_set_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvalRunnerError(f"invalid golden test set JSON: {test_set_path}") from exc
    raw_cases = payload.get("test_cases") if isinstance(payload, dict) else None
    if not isinstance(raw_cases, list):
        raise EvalRunnerError("golden test set must contain a test_cases list")
    return [_case_from_mapping(item, index) for index, item in enumerate(raw_cases, start=1)]


def _case_from_mapping(item: Any, index: int) -> GoldenTestCase:
    if not isinstance(item, Mapping):
        raise EvalRunnerError(f"test case #{index} must be an object")
    query = _required_text(item, "query", index)
    expected_chunk_ids = _string_list(item.get("expected_chunk_ids"))
    expected_sources = _string_list(item.get("expected_sources"))
    if not expected_chunk_ids and not expected_sources:
        raise EvalRunnerError(f"test case #{index} requires expected_chunk_ids or expected_sources")
    filters = item.get("filters") if isinstance(item.get("filters"), dict) else {}
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    top_k = item.get("top_k")
    return GoldenTestCase(
        query=query,
        expected_chunk_ids=expected_chunk_ids,
        expected_sources=expected_sources,
        answer=_optional_text(item.get("answer") or item.get("generated_answer")),
        ground_truth=_optional_text(item.get("ground_truth") or item.get("expected_answer")),
        contexts=_string_list(item.get("contexts")),
        filters=dict(filters),
        top_k=int(top_k) if top_k is not None else None,
        metadata=dict(metadata),
    )


def _aggregate_metrics(results: list[EvalCaseResult]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for result in results:
        for name, value in result.metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value)
            counts[name] = counts.get(name, 0) + 1
    metrics = {name: totals[name] / counts[name] for name in sorted(totals)}
    metrics["case_count"] = float(len(results))
    return metrics


def _retrieved_ids(results: list[RetrievalResult]) -> list[str]:
    ids: list[str] = []
    for result in results:
        ids.append(result.chunk_id)
        source_path = result.metadata.get("source_path") or result.metadata.get("source")
        if isinstance(source_path, str) and source_path:
            ids.append(source_path)
            ids.append(Path(source_path).name)
    return _unique(ids)


def _default_answer(results: list[RetrievalResult]) -> str:
    if not results:
        return "No answer generated."
    return "\n\n".join(result.text for result in results[:3])


def _validate_results(results: Any) -> None:
    if not isinstance(results, list) or not all(isinstance(result, RetrievalResult) for result in results):
        raise EvalRunnerError("hybrid_search.search must return list[RetrievalResult]")


def _required_text(item: Mapping[str, Any], key: str, index: int) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvalRunnerError(f"test case #{index} requires non-empty {key}")
    return value.strip()


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _record_trace(trace: TraceContext | None, name: str, data: dict[str, Any]) -> None:
    if hasattr(trace, "record_stage"):
        trace.record_stage(name, data)


__all__ = [
    "EvalCaseResult",
    "EvalReport",
    "EvalRunner",
    "EvalRunnerError",
    "GoldenTestCase",
    "load_golden_test_set",
]
