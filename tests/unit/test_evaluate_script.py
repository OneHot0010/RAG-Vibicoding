"""Tests for the evaluation CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.evaluate as evaluate_script
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


class FakeHybridSearch:
    def search(self, query: str, top_k: int, filters: dict[str, Any] | None = None, trace: Any | None = None) -> list[RetrievalResult]:
        return [RetrievalResult("chunk-a", 1.0, "answer context", {"source_path": "docs/a.pdf"})]


class FakeComponents:
    hybrid_search = FakeHybridSearch()


class FakeEvaluator(BaseEvaluator):
    def evaluate(self, case: EvaluationCase, trace: Any | None = None) -> EvaluationResult:
        return EvaluationResult(metrics={"hit_rate": 1.0}, details={"query": case.query})


def settings(path: Path) -> Settings:
    return Settings(
        llm=LLMSettings(provider="fake", model="fake-chat"),
        embedding=EmbeddingSettings(provider="fake", model="fake-embedding"),
        vision_llm=VisionLLMSettings(provider="fake", model="fake-vision"),
        splitter=SplitterSettings(strategy="fake", chunk_size=100, chunk_overlap=10),
        vector_store=VectorStoreSettings(backend="fake", persist_path="./tmp/vector"),
        retrieval=RetrievalSettings("bm25", "rrf", 20, 20, 3),
        rerank=RerankSettings(backend="none"),
        evaluation=EvaluationSettings(backends=["custom"], golden_test_set=str(path)),
        observability=ObservabilitySettings(enabled=True, log_file="./logs/traces.jsonl"),
    )


def test_evaluate_script_outputs_report(monkeypatch, tmp_path: Path, capsys) -> None:
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps({"test_cases": [{"query": "q", "expected_chunk_ids": ["chunk-a"]}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluate_script, "load_settings", lambda config: settings(golden))
    monkeypatch.setattr(evaluate_script, "build_query_components", lambda settings, data_dir, offline_embedding: FakeComponents())
    monkeypatch.setattr(evaluate_script.EvaluatorFactory, "create", lambda settings, backend=None: FakeEvaluator())

    exit_code = evaluate_script.main(["--config", "fake.yaml"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metrics"]["hit_rate"] == 1.0
    assert payload["case_count"] == 1
