"""Smoke tests for OpenAI and Azure embedding providers with mocked HTTP."""

from __future__ import annotations

from typing import Any
from urllib.error import URLError

import pytest

from core.settings import EmbeddingSettings
from libs.embedding import (
    AzureEmbedding,
    EmbeddingFactory,
    EmbeddingProviderError,
    OpenAIEmbedding,
)


@pytest.fixture(autouse=True)
def register_default_providers() -> None:
    EmbeddingFactory.clear()
    EmbeddingFactory.register("openai", OpenAIEmbedding)
    EmbeddingFactory.register("azure", AzureEmbedding)


def test_openai_embedding_posts_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(
        self: OpenAIEmbedding,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        captured.update(url=url, payload=payload, headers=headers)
        return {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]}

    monkeypatch.setattr(OpenAIEmbedding, "_post_json", fake_post)
    embedding = EmbeddingFactory.create(
        EmbeddingSettings(provider="openai", model="text-embedding-3-small", api_key="secret")
    )

    vectors = embedding.embed(["alpha", "beta"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert captured["url"] == "https://api.openai.com/v1/embeddings"
    assert captured["payload"] == {
        "model": "text-embedding-3-small",
        "input": ["alpha", "beta"],
    }
    assert captured["headers"]["Authorization"] == "Bearer secret"


def test_azure_embedding_uses_deployment_url_and_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(
        self: AzureEmbedding,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        captured.update(url=url, payload=payload, headers=headers)
        return {"data": [{"embedding": [1, 2, 3]}]}

    monkeypatch.setattr(AzureEmbedding, "_post_json", fake_post)
    embedding = EmbeddingFactory.create(
        EmbeddingSettings(
            provider="azure",
            model="text-embedding-ada-002",
            deployment="embed-prod",
            azure_endpoint="https://example.openai.azure.com",
            api_key="azure-secret",
            api_version="2024-10-21",
        )
    )

    vectors = embedding.embed(["hello"])

    assert vectors == [[1.0, 2.0, 3.0]]
    assert captured["url"] == (
        "https://example.openai.azure.com/openai/deployments/embed-prod/"
        "embeddings?api-version=2024-10-21"
    )
    assert captured["payload"] == {"input": ["hello"]}
    assert captured["headers"]["api-key"] == "azure-secret"


def test_factory_routes_builtin_embedding_providers() -> None:
    assert isinstance(
        EmbeddingFactory.create(
            EmbeddingSettings(provider="openai", model="text-embedding-3-small")
        ),
        OpenAIEmbedding,
    )
    assert isinstance(
        EmbeddingFactory.create(
            EmbeddingSettings(
                provider="azure",
                model="text-embedding-ada-002",
                azure_endpoint="https://example.openai.azure.com",
            )
        ),
        AzureEmbedding,
    )


def test_azure_embedding_requires_endpoint() -> None:
    embedding = AzureEmbedding(EmbeddingSettings(provider="azure", model="text-embedding-ada-002"))

    with pytest.raises(EmbeddingProviderError, match="azure embedding requires"):
        embedding.embed(["hello"])


def test_empty_input_has_clear_error() -> None:
    embedding = OpenAIEmbedding(EmbeddingSettings(provider="openai", model="text-embedding-3-small"))

    with pytest.raises(EmbeddingProviderError, match="openai texts must not be empty"):
        embedding.embed([])


def test_invalid_text_item_has_clear_error() -> None:
    embedding = OpenAIEmbedding(EmbeddingSettings(provider="openai", model="text-embedding-3-small"))

    with pytest.raises(EmbeddingProviderError, match=r"openai texts\[1\]"):
        embedding.embed(["hello", ""])  # empty string is invalid


def test_response_count_mismatch_has_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(
        self: OpenAIEmbedding,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        return {"data": [{"embedding": [0.1]}]}

    monkeypatch.setattr(OpenAIEmbedding, "_post_json", fake_post)
    embedding = OpenAIEmbedding(EmbeddingSettings(provider="openai", model="text-embedding-3-small"))

    with pytest.raises(EmbeddingProviderError, match="embedding count mismatch"):
        embedding.embed(["a", "b"])


def test_connection_error_is_readable_and_does_not_leak_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*args: Any, **kwargs: Any) -> Any:
        raise URLError("secret-key should not be exposed")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    embedding = OpenAIEmbedding(
        EmbeddingSettings(
            provider="openai",
            model="text-embedding-3-small",
            api_key="secret-key",
        )
    )

    with pytest.raises(EmbeddingProviderError) as exc_info:
        embedding.embed(["hello"])

    message = str(exc_info.value)
    assert "openai request failed" in message
    assert "secret-key" not in message
