"""Tests for splitter abstraction and factory routing."""

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
from libs.splitter import BaseSplitter, SplitterFactory, SplitterFactoryError


class FakeSplitter(BaseSplitter):
    def __init__(self, settings: SplitterSettings) -> None:
        self.settings = settings

    def split_text(self, text: str, trace: Any | None = None) -> list[str]:
        if not text:
            return []
        return [part.strip() for part in text.split("|") if part.strip()]


class NotASplitter:
    pass


@pytest.fixture(autouse=True)
def clear_factory_registry() -> None:
    SplitterFactory.clear()


def make_settings(strategy: str = "fake") -> Settings:
    return Settings(
        llm=LLMSettings(provider="fake", model="fake-chat"),
        embedding=EmbeddingSettings(provider="fake", model="fake-embedding"),
        vision_llm=VisionLLMSettings(provider="fake", model="fake-vision"),
        splitter=SplitterSettings(strategy=strategy, chunk_size=100, chunk_overlap=10),
        vector_store=VectorStoreSettings(backend="fake", persist_path="./tmp/vector"),
        retrieval=RetrievalSettings(
            sparse_backend="bm25",
            fusion_algorithm="rrf",
            top_k_dense=20,
            top_k_sparse=20,
            top_k_final=10,
        ),
        rerank=RerankSettings(backend="none"),
        evaluation=EvaluationSettings(backends=["custom"], golden_test_set="./golden.json"),
        observability=ObservabilitySettings(enabled=True, log_file="./logs/traces.jsonl"),
    )


def test_base_splitter_split_text_contract() -> None:
    splitter = FakeSplitter(SplitterSettings(strategy="fake", chunk_size=100, chunk_overlap=10))

    chunks = splitter.split_text("alpha | beta | gamma")

    assert chunks == ["alpha", "beta", "gamma"]


def test_factory_routes_using_full_settings() -> None:
    SplitterFactory.register("fake", FakeSplitter)

    splitter = SplitterFactory.create(make_settings(strategy="fake"))

    assert isinstance(splitter, FakeSplitter)
    assert splitter.settings.chunk_size == 100


def test_factory_routes_using_splitter_settings_directly() -> None:
    SplitterFactory.register("fake", FakeSplitter)

    splitter = SplitterFactory.create(SplitterSettings(strategy="fake", chunk_size=256, chunk_overlap=32))

    assert isinstance(splitter, FakeSplitter)
    assert splitter.settings.chunk_overlap == 32


def test_factory_normalizes_strategy_names() -> None:
    SplitterFactory.register("fake", FakeSplitter)

    splitter = SplitterFactory.create(SplitterSettings(strategy=" FAKE ", chunk_size=100, chunk_overlap=10))

    assert isinstance(splitter, FakeSplitter)


def test_factory_unknown_strategy_has_readable_error() -> None:
    SplitterFactory.register("fake", FakeSplitter)

    with pytest.raises(SplitterFactoryError, match="Unknown splitter strategy: semantic"):
        SplitterFactory.create(SplitterSettings(strategy="semantic", chunk_size=100, chunk_overlap=10))


def test_register_requires_callable_builder() -> None:
    with pytest.raises(TypeError, match="builder must be callable"):
        SplitterFactory.register("bad", "not-callable")  # type: ignore[arg-type]


def test_factory_rejects_builder_returning_wrong_type() -> None:
    SplitterFactory.register("bad", lambda settings: NotASplitter())  # type: ignore[return-value]

    with pytest.raises(SplitterFactoryError, match="expected BaseSplitter"):
        SplitterFactory.create(SplitterSettings(strategy="bad", chunk_size=100, chunk_overlap=10))


def test_unregister_removes_strategy() -> None:
    SplitterFactory.register("fake", FakeSplitter)
    SplitterFactory.unregister("fake")

    with pytest.raises(SplitterFactoryError, match="Registered strategies: none"):
        SplitterFactory.create(SplitterSettings(strategy="fake", chunk_size=100, chunk_overlap=10))
