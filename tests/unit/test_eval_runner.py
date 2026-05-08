"""Tests for golden-set evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.settings import (
    EmbeddingSettings,
    EvaluationSettings,
    LLMSettings,
    ObservabilitySettings,
    RerankSettings,
    RetrievalSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
    VisionLLMSettings,
)
from core.trace import TraceContext
from core.types import RetrievalResult
from libs.evaluator import BaseEvaluator, EvaluationCase, EvaluationResult
from observability.evaluation import EvalRunner, EvalRunnerError, load_golden_test_set


class FakeHybridSearch:
    def __init__(self, results_by_query: dict[str, list[RetrievalResult]]) -> None:
        self.results_by_query = results_by_query
        self.calls: list[tuple[str, int, dict[str, Any]]] = []

    def search(self, query: str, top_k: int, filters: dict[str, Any] | None = None, trace: Any | None = None) -> list[RetrievalResult]:
        self.calls.append((query, top_k, dict(filters or {})))
        return self.results_by_query.get(query, [])[:top_k]


class EchoEvaluator(BaseEvaluator):
    def evaluate(self, case: EvaluationCase, trace: Any | None = None) -> EvaluationResult:
        hits = [item for item in case.retrieved_ids if item in case.golden_ids]
        return EvaluationResult(
            metrics={"hit_rate": 1.0 if hits else 0.0, "retrieved_count": float(len(case.retrieved_ids))},
            details={"matched_ids": hits, "answer": case.metadata.get("answer")},
        )


def make_settings() -> Settings:
    return Settings(
        llm=LLMSettings(provider="fake", model="fake-chat"),
        embedding=EmbeddingSettings(provider="fake", model="fake-embedding"),
        vision_llm=VisionLLMSettings(provider="fake", model="fake-vision"),
        splitter=SplitterSettings(strategy="fake", chunk_size=100, chunk_overlap=10),
        vector_store=VectorStoreSettings(backend="fake", persist_path="./tmp/vector"),
        retrieval=RetrievalSettings("bm25", "rrf", 20, 20, 3),
        rerank=RerankSettings(backend="none"),
        evaluation=EvaluationSettings(backends=["custom"], golden_test_set="./golden.json"),
        observability=ObservabilitySettings(enabled=True, log_file="./logs/traces.jsonl"),
    )


def write_golden(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "test_cases": [
                    {
                        "query": "azure config",
                        "expected_chunk_ids": ["chunk-a"],
                        "expected_sources": ["docs/a.pdf"],
                        "expected_answer": "configure azure",
                        "filters": {"collection": "docs"},
                        "top_k": 2,
                    },
                    {
                        "query": "hybrid retrieval",
                        "expected_sources": ["retrieval.pdf"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def result(chunk_id: str, source_path: str, text: str = "context") -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, score=0.9, text=text, metadata={"source_path": source_path})


def test_load_golden_test_set_accepts_chunk_ids_and_sources(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    write_golden(path)

    cases = load_golden_test_set(path)

    assert cases[0].query == "azure config"
    assert cases[0].golden_ids() == ["chunk-a", "docs/a.pdf"]
    assert cases[0].filters == {"collection": "docs"}
    assert cases[1].golden_ids() == ["retrieval.pdf"]


def test_eval_runner_runs_cases_and_aggregates_metrics(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    write_golden(path)
    hybrid = FakeHybridSearch(
        {
            "azure config": [result("chunk-a", "docs/a.pdf", "azure context")],
            "hybrid retrieval": [result("chunk-x", "docs/retrieval.pdf", "retrieval context")],
        }
    )
    trace = TraceContext(trace_type="query")

    report = EvalRunner(make_settings(), hybrid_search=hybrid, evaluator=EchoEvaluator()).run(path, trace=trace)

    assert report.metrics["hit_rate"] == 1.0
    assert report.metrics["case_count"] == 2.0
    assert report.cases[0].retrieved_ids == ["chunk-a", "docs/a.pdf", "a.pdf"]
    assert report.cases[1].golden_ids == ["retrieval.pdf"]
    assert hybrid.calls == [("azure config", 2, {"collection": "docs"}), ("hybrid retrieval", 3, {})]
    assert trace.stages[-1]["name"] == "eval_runner.completed"


def test_eval_runner_rejects_invalid_test_sets(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(EvalRunnerError, match="not found"):
        load_golden_test_set(missing)

    empty = tmp_path / "empty.json"
    empty.write_text('{"test_cases": []}', encoding="utf-8")
    with pytest.raises(EvalRunnerError, match="no test_cases"):
        EvalRunner(make_settings(), FakeHybridSearch({}), EchoEvaluator()).run(empty)

    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"test_cases": [{"query": "q"}]}', encoding="utf-8")
    with pytest.raises(EvalRunnerError, match="expected_chunk_ids"):
        load_golden_test_set(invalid)


def test_eval_report_serializes_to_dict(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    write_golden(path)
    report = EvalRunner(
        make_settings(),
        hybrid_search=FakeHybridSearch({"azure config": [result("chunk-a", "docs/a.pdf")]}),
        evaluator=EchoEvaluator(),
    ).run(path)

    payload = report.to_dict()

    assert payload["case_count"] == 2
    assert payload["cases"][0]["query"] == "azure config"
    assert "hit_rate" in payload["metrics"]
