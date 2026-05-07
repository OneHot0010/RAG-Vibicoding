"""Tests for the embedding abstraction and factory."""

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
    VectorStoreSettings,
    VisionLLMSettings,
)
from libs.embedding import BaseEmbedding, EmbeddingFactory, EmbeddingFactoryError


class FakeEmbedding(BaseEmbedding):
    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        return [[float(len(text)), float(sum(ord(char) for char in text) % 100)] for text in texts]


class NotAnEmbedding:
    pass


@pytest.fixture(autouse=True)
def clear_factory_registry() -> None:
    EmbeddingFactory.clear()


def make_settings(provider: str = "fake") -> Settings:
    return Settings(
        llm=LLMSettings(provider="fake", model="fake-chat"),
        embedding=EmbeddingSettings(provider=provider, model="fake-embedding"),
        vision_llm=VisionLLMSettings(provider="fake", model="fake-vision"),
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


def test_base_embedding_embed_contract_is_stable() -> None:
    embedding = FakeEmbedding(EmbeddingSettings(provider="fake", model="fake-embedding"))

    vectors = embedding.embed(["alpha", "beta", "alpha"])

    assert vectors[0] == vectors[2]
    assert len(vectors) == 3
    assert all(len(vector) == 2 for vector in vectors)


def test_factory_routes_using_full_settings() -> None:
    EmbeddingFactory.register("fake", FakeEmbedding)

    embedding = EmbeddingFactory.create(make_settings(provider="fake"))

    assert isinstance(embedding, FakeEmbedding)
    assert embedding.settings.model == "fake-embedding"


def test_factory_routes_using_embedding_settings_directly() -> None:
    EmbeddingFactory.register("fake", FakeEmbedding)

    embedding = EmbeddingFactory.create(EmbeddingSettings(provider="fake", model="direct-model"))

    assert isinstance(embedding, FakeEmbedding)
    assert embedding.settings.model == "direct-model"


def test_factory_normalizes_provider_names() -> None:
    EmbeddingFactory.register("fake", FakeEmbedding)

    embedding = EmbeddingFactory.create(EmbeddingSettings(provider=" FAKE ", model="fake-embedding"))

    assert isinstance(embedding, FakeEmbedding)


def test_factory_unknown_provider_has_readable_error() -> None:
    EmbeddingFactory.register("fake", FakeEmbedding)

    with pytest.raises(EmbeddingFactoryError, match="Unknown embedding provider: missing"):
        EmbeddingFactory.create(EmbeddingSettings(provider="missing", model="fake-embedding"))


def test_register_requires_callable_builder() -> None:
    with pytest.raises(TypeError, match="builder must be callable"):
        EmbeddingFactory.register("bad", "not-callable")  # type: ignore[arg-type]


def test_factory_rejects_builder_returning_wrong_type() -> None:
    EmbeddingFactory.register("bad", lambda settings: NotAnEmbedding())  # type: ignore[return-value]

    with pytest.raises(EmbeddingFactoryError, match="expected BaseEmbedding"):
        EmbeddingFactory.create(EmbeddingSettings(provider="bad", model="fake-embedding"))


def test_unregister_removes_provider() -> None:
    EmbeddingFactory.register("fake", FakeEmbedding)
    EmbeddingFactory.unregister("fake")

    with pytest.raises(EmbeddingFactoryError, match="Registered providers: none"):
        EmbeddingFactory.create(EmbeddingSettings(provider="fake", model="fake-embedding"))
