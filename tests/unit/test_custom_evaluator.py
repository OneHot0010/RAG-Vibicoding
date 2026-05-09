"""Tests for the custom evaluator and evaluator factory."""

from __future__ import annotations

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
from libs.evaluator import (
    BaseEvaluator,
    CustomEvaluator,
    EvaluationCase,
    EvaluationResult,
    EvaluatorFactory,
    EvaluatorFactoryError,
)


class FakeEvaluator(BaseEvaluator):
    def __init__(self, settings: EvaluationSettings) -> None:
        self.settings = settings

    def evaluate(self, case: EvaluationCase, trace: Any | None = None) -> EvaluationResult:
        return EvaluationResult(metrics={"fake_score": 1.0}, details={"query": case.query})


class NotAnEvaluator:
    pass


@pytest.fixture(autouse=True)
def reset_factory_registry() -> None:
    EvaluatorFactory.reset_defaults()


def make_settings(backends: list[str] | None = None) -> Settings:
    return Settings(
        llm=LLMSettings(provider="fake", model="fake-chat"),
        embedding=EmbeddingSettings(provider="fake", model="fake-embedding"),
        vision_llm=VisionLLMSettings(provider="fake", model="fake-vision"),
        splitter=SplitterSettings(strategy="fake", chunk_size=100, chunk_overlap=10),
        vector_store=VectorStoreSettings(backend="fake", persist_path="./tmp/vector"),
        retrieval=RetrievalSettings(
            sparse_backend="bm25",
            fusion_algorithm="rrf",
            top_k_dense=20,
            top_k_sparse=20,
            top_k_final=10,
        ),
        rerank=RerankSettings(backend="none"),
        evaluation=EvaluationSettings(backends=backends or ["custom"], golden_test_set="./golden.json"),
        observability=ObservabilitySettings(enabled=True, log_file="./logs/traces.jsonl"),
    )


def test_custom_evaluator_hit_at_first_rank() -> None:
    result = CustomEvaluator().evaluate(
        EvaluationCase(
            query="config",
            retrieved_ids=["doc-a", "doc-b"],
            golden_ids=["doc-a"],
        )
    )

    assert result.metrics["hit_rate"] == 1.0
    assert result.metrics["mrr"] == 1.0
    assert result.details["first_hit_rank"] == 1
    assert result.details["matched_ids"] == ["doc-a"]


def test_custom_evaluator_hit_at_later_rank() -> None:
    result = CustomEvaluator().evaluate(
        EvaluationCase(
            query="config",
            retrieved_ids=["doc-x", "doc-a", "doc-b"],
            golden_ids=["doc-a", "doc-z"],
        )
    )

    assert result.metrics["hit_rate"] == 1.0
    assert result.metrics["mrr"] == 0.5
    assert result.details["first_hit_rank"] == 2


def test_custom_evaluator_no_hit() -> None:
    result = CustomEvaluator().evaluate(
        EvaluationCase(query="config", retrieved_ids=["doc-x"], golden_ids=["doc-a"])
    )

    assert result.metrics["hit_rate"] == 0.0
    assert result.metrics["mrr"] == 0.0
    assert result.details["first_hit_rank"] is None
    assert result.details["matched_ids"] == []


def test_custom_evaluator_empty_retrieved_ids_is_no_hit() -> None:
    result = CustomEvaluator().evaluate(
        EvaluationCase(query="config", retrieved_ids=[], golden_ids=["doc-a"])
    )

    assert result.metrics["hit_rate"] == 0.0
    assert result.metrics["mrr"] == 0.0
    assert result.metrics["retrieved_count"] == 0.0
    assert result.metrics["golden_count"] == 1.0
    assert result.details["matched_ids"] == []


def test_custom_evaluator_duplicate_retrieved_hits_preserve_detail_order() -> None:
    result = CustomEvaluator().evaluate(
        EvaluationCase(query="config", retrieved_ids=["doc-a", "doc-a", "doc-b"], golden_ids=["doc-a"])
    )

    assert result.metrics["hit_rate"] == 1.0
    assert result.metrics["mrr"] == 1.0
    assert result.details["matched_ids"] == ["doc-a", "doc-a"]


def test_custom_evaluator_empty_golden_set_is_no_hit() -> None:
    result = CustomEvaluator().evaluate(
        EvaluationCase(query="config", retrieved_ids=["doc-x"], golden_ids=[])
    )

    assert result.metrics["hit_rate"] == 0.0
    assert result.metrics["mrr"] == 0.0
    assert result.metrics["retrieved_count"] == 1.0
    assert result.metrics["golden_count"] == 0.0


def test_factory_uses_first_configured_backend() -> None:
    evaluator = EvaluatorFactory.create(make_settings(backends=["custom"]))

    assert isinstance(evaluator, CustomEvaluator)


def test_factory_routes_explicit_backend() -> None:
    EvaluatorFactory.register("fake", FakeEvaluator)

    evaluator = EvaluatorFactory.create(make_settings(backends=["custom", "fake"]), backend="fake")

    assert isinstance(evaluator, FakeEvaluator)
    assert evaluator.settings.backends == ["custom", "fake"]


def test_factory_normalizes_backend_names() -> None:
    EvaluatorFactory.register("fake", FakeEvaluator)

    evaluator = EvaluatorFactory.create(
        EvaluationSettings(backends=[" FAKE "], golden_test_set="./golden.json")
    )

    assert isinstance(evaluator, FakeEvaluator)


def test_factory_unknown_backend_has_readable_error() -> None:
    with pytest.raises(EvaluatorFactoryError, match="Unknown evaluator backend: missing"):
        EvaluatorFactory.create(
            EvaluationSettings(backends=["missing"], golden_test_set="./golden.json")
        )


@pytest.mark.parametrize("backend", ["", "   "])
def test_factory_rejects_blank_explicit_backend_name(backend: str) -> None:
    with pytest.raises(EvaluatorFactoryError, match="backend name is required"):
        EvaluatorFactory.create(EvaluationSettings(backends=["custom"], golden_test_set="./golden.json"), backend=backend)


def test_factory_requires_at_least_one_backend() -> None:
    with pytest.raises(EvaluatorFactoryError, match="At least one evaluator backend"):
        EvaluatorFactory.create(EvaluationSettings(backends=[], golden_test_set="./golden.json"))


def test_register_requires_callable_builder() -> None:
    with pytest.raises(TypeError, match="builder must be callable"):
        EvaluatorFactory.register("bad", "not-callable")  # type: ignore[arg-type]


def test_factory_rejects_builder_returning_wrong_type() -> None:
    EvaluatorFactory.register("bad", lambda settings: NotAnEvaluator())  # type: ignore[return-value]

    with pytest.raises(EvaluatorFactoryError, match="expected BaseEvaluator"):
        EvaluatorFactory.create(EvaluationSettings(backends=["bad"], golden_test_set="./golden.json"))


def test_unregister_restores_custom_backend_default() -> None:
    EvaluatorFactory.register("custom", FakeEvaluator)
    EvaluatorFactory.unregister("custom")

    evaluator = EvaluatorFactory.create(EvaluationSettings(backends=["custom"], golden_test_set="./golden.json"))

    assert isinstance(evaluator, CustomEvaluator)
