"""Tests for the LLM abstraction and registry-backed factory."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

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
from libs.llm import BaseLLM, ChatMessage, LLMFactory, LLMFactoryError


class FakeLLM(BaseLLM):
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    def chat(self, messages: Sequence[ChatMessage | Mapping[str, Any]]) -> str:
        return f"{self.settings.provider}:{len(messages)}"


class NotAnLLM:
    pass


@pytest.fixture(autouse=True)
def clear_factory_registry() -> None:
    LLMFactory.clear()


def make_settings(provider: str = "fake") -> Settings:
    return Settings(
        llm=LLMSettings(provider=provider, model="fake-chat"),
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
        evaluation=EvaluationSettings(backends=["custom"], golden_test_set="./golden.json"),
        observability=ObservabilitySettings(enabled=True, log_file="./logs/traces.jsonl"),
    )


def test_base_llm_chat_contract() -> None:
    llm = FakeLLM(LLMSettings(provider="fake", model="fake-chat"))

    response = llm.chat([ChatMessage(role="user", content="hello")])

    assert response == "fake:1"


def test_factory_routes_using_full_settings() -> None:
    LLMFactory.register("fake", FakeLLM)

    llm = LLMFactory.create(make_settings(provider="fake"))

    assert isinstance(llm, FakeLLM)
    assert llm.settings.model == "fake-chat"


def test_factory_routes_using_llm_settings_directly() -> None:
    LLMFactory.register("fake", FakeLLM)

    llm = LLMFactory.create(LLMSettings(provider="fake", model="direct-model"))

    assert isinstance(llm, FakeLLM)
    assert llm.settings.model == "direct-model"


def test_factory_normalizes_provider_names() -> None:
    LLMFactory.register("fake", FakeLLM)

    llm = LLMFactory.create(LLMSettings(provider=" FAKE ", model="fake-chat"))

    assert isinstance(llm, FakeLLM)


def test_factory_unknown_provider_has_readable_error() -> None:
    LLMFactory.register("fake", FakeLLM)

    with pytest.raises(LLMFactoryError, match="Unknown LLM provider: missing"):
        LLMFactory.create(LLMSettings(provider="missing", model="fake-chat"))


def test_register_requires_callable_builder() -> None:
    with pytest.raises(TypeError, match="builder must be callable"):
        LLMFactory.register("bad", "not-callable")  # type: ignore[arg-type]


def test_factory_rejects_builder_returning_wrong_type() -> None:
    LLMFactory.register("bad", lambda settings: NotAnLLM())  # type: ignore[return-value]

    with pytest.raises(LLMFactoryError, match="expected BaseLLM"):
        LLMFactory.create(LLMSettings(provider="bad", model="fake-chat"))


def test_unregister_removes_provider() -> None:
    LLMFactory.register("fake", FakeLLM)
    LLMFactory.unregister("fake")

    with pytest.raises(LLMFactoryError, match="Registered providers: none"):
        LLMFactory.create(LLMSettings(provider="fake", model="fake-chat"))
