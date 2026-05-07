"""Tests for the Vision LLM abstraction and factory routing."""

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
from libs.llm import BaseVisionLLM, ChatResponse, LLMFactory, LLMFactoryError


class FakeVisionLLM(BaseVisionLLM):
    def __init__(self, settings: VisionLLMSettings) -> None:
        self.settings = settings
        self.calls: list[tuple[str, str | bytes, Any | None]] = []

    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any | None = None,
    ) -> ChatResponse:
        self.calls.append((text, image_path, trace))
        return ChatResponse(
            content=f"{self.settings.provider}:{self.settings.model}:{text}",
            metadata={"image_type": type(image_path).__name__},
        )


class NotAVisionLLM:
    pass


@pytest.fixture(autouse=True)
def clear_factory_registry() -> None:
    LLMFactory.clear()


def make_settings(provider: str = "fake") -> Settings:
    return Settings(
        llm=LLMSettings(provider="fake", model="fake-chat"),
        embedding=EmbeddingSettings(provider="fake", model="fake-embedding"),
        vision_llm=VisionLLMSettings(provider=provider, model="fake-vision"),
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


def test_base_vision_llm_chat_with_image_contract() -> None:
    vision_llm = FakeVisionLLM(VisionLLMSettings(provider="fake", model="fake-vision"))

    response = vision_llm.chat_with_image("describe", b"image-bytes")

    assert response == ChatResponse(
        content="fake:fake-vision:describe",
        metadata={"image_type": "bytes"},
    )
    assert vision_llm.calls == [("describe", b"image-bytes", None)]


def test_create_vision_llm_routes_using_full_settings() -> None:
    LLMFactory.register_vision("fake", FakeVisionLLM)

    vision_llm = LLMFactory.create_vision_llm(make_settings(provider="fake"))

    assert isinstance(vision_llm, FakeVisionLLM)
    assert vision_llm.settings.model == "fake-vision"


def test_create_vision_llm_routes_using_vision_settings_directly() -> None:
    LLMFactory.register_vision("fake", FakeVisionLLM)

    vision_llm = LLMFactory.create_vision_llm(
        VisionLLMSettings(provider="fake", model="direct-vision")
    )

    assert isinstance(vision_llm, FakeVisionLLM)
    assert vision_llm.settings.model == "direct-vision"


def test_create_vision_llm_normalizes_provider_names() -> None:
    LLMFactory.register_vision("fake", FakeVisionLLM)

    vision_llm = LLMFactory.create_vision_llm(
        VisionLLMSettings(provider=" FAKE ", model="fake-vision")
    )

    assert isinstance(vision_llm, FakeVisionLLM)


def test_unknown_vision_provider_has_readable_error() -> None:
    LLMFactory.register_vision("fake", FakeVisionLLM)

    with pytest.raises(LLMFactoryError, match="Unknown Vision LLM provider: missing"):
        LLMFactory.create_vision_llm(VisionLLMSettings(provider="missing", model="fake-vision"))


def test_register_vision_requires_callable_builder() -> None:
    with pytest.raises(TypeError, match="builder must be callable"):
        LLMFactory.register_vision("bad", "not-callable")  # type: ignore[arg-type]


def test_create_vision_llm_rejects_builder_returning_wrong_type() -> None:
    LLMFactory.register_vision("bad", lambda settings: NotAVisionLLM())  # type: ignore[return-value]

    with pytest.raises(LLMFactoryError, match="expected BaseVisionLLM"):
        LLMFactory.create_vision_llm(VisionLLMSettings(provider="bad", model="fake-vision"))


def test_unregister_vision_removes_provider() -> None:
    LLMFactory.register_vision("fake", FakeVisionLLM)
    LLMFactory.unregister_vision("fake")

    with pytest.raises(LLMFactoryError, match="Registered providers: none"):
        LLMFactory.create_vision_llm(VisionLLMSettings(provider="fake", model="fake-vision"))
