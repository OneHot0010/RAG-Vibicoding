"""Tests for dashboard evaluation panel model and service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
from core.types import RetrievalResult
from libs.evaluator import BaseEvaluator, EvaluationCase, EvaluationResult
from observability.dashboard.pages.evaluation_panel import evaluation_panel_model, render
from observability.dashboard.services.evaluation_service import EvaluationService


class FakeEvaluationService:
    def available_backends(self) -> list[str]:
        return ["all", "custom", "ragas"]

    def default_test_set_path(self) -> str:
        return "tests/fixtures/golden_test_set.json"

    def test_set_summary(self, test_set_path: str | None = None) -> dict[str, Any]:
        return {"path": test_set_path or self.default_test_set_path(), "case_count": 2, "queries": ["q1", "q2"]}

    def run_evaluation(self, **kwargs: Any) -> dict[str, Any]:
        return report()


class FakeHybridSearch:
    def search(self, query: str, top_k: int, filters: dict[str, Any] | None = None, trace: Any | None = None) -> list[RetrievalResult]:
        return [RetrievalResult("chunk-a", 1.0, "context", {"source_path": "docs/a.pdf"})]


class FakeComponents:
    hybrid_search = FakeHybridSearch()


class FakeEvaluator(BaseEvaluator):
    def evaluate(self, case: EvaluationCase, trace: Any | None = None) -> EvaluationResult:
        return EvaluationResult(metrics={"hit_rate": 1.0, "mrr": 1.0}, details={"query": case.query})


class FakeEvaluatorFactory:
    @staticmethod
    def create(settings: Settings, backend: str | None = None) -> BaseEvaluator:
        return FakeEvaluator()


def settings(path: Path) -> Settings:
    return Settings(
        llm=LLMSettings(provider="fake", model="fake-chat"),
        embedding=EmbeddingSettings(provider="fake", model="fake-embedding"),
        vision_llm=VisionLLMSettings(provider="fake", model="fake-vision"),
        splitter=SplitterSettings(strategy="fake", chunk_size=100, chunk_overlap=10),
        vector_store=VectorStoreSettings(backend="fake", persist_path="./tmp/vector"),
        retrieval=RetrievalSettings("bm25", "rrf", 20, 20, 3),
        rerank=RerankSettings(backend="none"),
        evaluation=EvaluationSettings(backends=["custom", "ragas"], golden_test_set=str(path)),
        observability=ObservabilitySettings(enabled=True, log_file="./logs/traces.jsonl"),
    )


def report() -> dict[str, Any]:
    return {
        "metrics": {"hit_rate": 1.0, "mrr": 0.5, "case_count": 2.0},
        "case_count": 2,
        "test_set_path": "golden.json",
        "cases": [
            {
                "query": "q1",
                "metrics": {"hit_rate": 1.0, "mrr": 1.0},
                "retrieved_ids": ["chunk-a"],
                "golden_ids": ["chunk-a"],
                "details": {"matched_ids": ["chunk-a"]},
            },
            {
                "query": "q2",
                "metrics": {"hit_rate": 0.0, "mrr": 0.0},
                "retrieved_ids": ["chunk-b"],
                "golden_ids": ["chunk-a"],
                "details": {"matched_ids": []},
            },
        ],
    }


def write_settings(path: Path, golden: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
llm:
  provider: fake
  model: fake-chat
embedding:
  provider: fake
  model: fake-embedding
vision_llm:
  provider: fake
  model: fake-vision
splitter:
  strategy: recursive
  chunk_size: 100
  chunk_overlap: 10
vector_store:
  backend: chroma
  persist_path: ./data/db/chroma
retrieval:
  sparse_backend: bm25
  fusion_algorithm: rrf
  top_k_dense: 4
  top_k_sparse: 5
  top_k_final: 3
rerank:
  backend: none
evaluation:
  backends:
    - custom
    - ragas
  golden_test_set: {golden.as_posix()}
observability:
  enabled: true
  log_file: ./logs/traces.jsonl
""".strip(),
        encoding="utf-8",
    )


def write_golden(path: Path) -> None:
    path.write_text(json.dumps({"test_cases": [{"query": "q", "expected_chunk_ids": ["chunk-a"]}]}), encoding="utf-8")


def test_evaluation_panel_model_formats_metrics_and_cases() -> None:
    model = evaluation_panel_model(FakeEvaluationService(), last_report=report())

    assert model["backends"] == ["all", "custom", "ragas"]
    assert model["test_set"]["case_count"] == 2
    assert model["metric_rows"] == [
        {"metric": "case_count", "value": 2.0},
        {"metric": "hit_rate", "value": 1.0},
        {"metric": "mrr", "value": 0.5},
    ]
    assert model["case_rows"][0] == {"query": "q1", "hit_rate": 1.0, "mrr": 1.0, "retrieved": 1, "golden": 1}


def test_evaluation_panel_render_without_streamlit_returns_model() -> None:
    rendered = render(evaluation_service=FakeEvaluationService())

    assert rendered["test_set"]["path"] == "tests/fixtures/golden_test_set.json"


def test_evaluation_service_runs_eval_runner(tmp_path: Path, monkeypatch) -> None:
    golden = tmp_path / "golden.json"
    settings_path = tmp_path / "config" / "settings.yaml"
    write_golden(golden)
    write_settings(settings_path, golden)
    service = EvaluationService(
        settings_path=settings_path,
        data_dir=tmp_path / "data",
        query_components_builder=lambda settings, data_dir, offline_embedding: FakeComponents(),
        evaluator_factory=FakeEvaluatorFactory,
    )

    assert service.available_backends() == ["all", "custom", "ragas"]
    assert service.test_set_summary()["case_count"] == 1
    result = service.run_evaluation(backend="all", top_k=2)

    assert result["metrics"]["hit_rate"] == 1.0
    assert result["case_count"] == 1
