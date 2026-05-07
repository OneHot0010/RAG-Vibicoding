"""Smoke tests for Ollama embedding provider with mocked HTTP."""

from __future__ import annotations

from typing import Any
from urllib.error import URLError

import pytest

from core.settings import EmbeddingSettings
from libs.embedding import EmbeddingFactory, EmbeddingProviderError, OllamaEmbedding


@pytest.fixture(autouse=True)
def register_ollama_provider() -> None:
    EmbeddingFactory.clear()
    EmbeddingFactory.register("ollama", OllamaEmbedding)


def test_factory_creates_ollama_embedding_provider() -> None:
    embedding = EmbeddingFactory.create(EmbeddingSettings(provider="ollama", model="nomic-embed-text"))

    assert isinstance(embedding, OllamaEmbedding)


def test_ollama_embedding_posts_one_request_per_text(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, Any]] = []

    def fake_post(self: OllamaEmbedding, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured.append({"url": url, "payload": payload})
        if payload["prompt"] == "alpha":
            return {"embedding": [1, 2, 3]}
        return {"embedding": [4, 5, 6]}

    monkeypatch.setattr(OllamaEmbedding, "_post_json", fake_post)
    embedding = OllamaEmbedding(
        EmbeddingSettings(provider="ollama", model="nomic-embed-text", base_url="http://ollama.test:11434")
    )

    vectors = embedding.embed(["alpha", "beta"])

    assert vectors == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert captured == [
        {
            "url": "http://ollama.test:11434/api/embeddings",
            "payload": {"model": "nomic-embed-text", "prompt": "alpha"},
        },
        {
            "url": "http://ollama.test:11434/api/embeddings",
            "payload": {"model": "nomic-embed-text", "prompt": "beta"},
        },
    ]


def test_ollama_embedding_uses_localhost_default_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(self: OllamaEmbedding, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured["url"] = url
        return {"embedding": [0.1, 0.2]}

    monkeypatch.setattr(OllamaEmbedding, "_post_json", fake_post)
    embedding = OllamaEmbedding(EmbeddingSettings(provider="ollama", model="nomic-embed-text"))

    assert embedding.embed(["hello"]) == [[0.1, 0.2]]
    assert captured["url"] == "http://localhost:11434/api/embeddings"


def test_empty_input_has_clear_error() -> None:
    embedding = OllamaEmbedding(EmbeddingSettings(provider="ollama", model="nomic-embed-text"))

    with pytest.raises(EmbeddingProviderError, match="ollama texts must not be empty"):
        embedding.embed([])


def test_invalid_text_item_has_clear_error() -> None:
    embedding = OllamaEmbedding(EmbeddingSettings(provider="ollama", model="nomic-embed-text"))

    with pytest.raises(EmbeddingProviderError, match=r"ollama texts\[1\]"):
        embedding.embed(["hello", ""])


def test_response_shape_error_mentions_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(self: OllamaEmbedding, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"done": True}

    monkeypatch.setattr(OllamaEmbedding, "_post_json", fake_post)
    embedding = OllamaEmbedding(EmbeddingSettings(provider="ollama", model="nomic-embed-text"))

    with pytest.raises(EmbeddingProviderError, match="ollama response missing embedding list"):
        embedding.embed(["hello"])


def test_non_numeric_embedding_has_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(self: OllamaEmbedding, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"embedding": [1, "bad"]}

    monkeypatch.setattr(OllamaEmbedding, "_post_json", fake_post)
    embedding = OllamaEmbedding(EmbeddingSettings(provider="ollama", model="nomic-embed-text"))

    with pytest.raises(EmbeddingProviderError, match="ollama response embedding must be numeric"):
        embedding.embed(["hello"])


def test_connection_error_is_readable_and_does_not_leak_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*args: Any, **kwargs: Any) -> Any:
        raise URLError("secret-key should not be exposed")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    embedding = OllamaEmbedding(
        EmbeddingSettings(
            provider="ollama",
            model="nomic-embed-text",
            base_url="http://localhost:11434",
            api_key="secret-key",
        )
    )

    with pytest.raises(EmbeddingProviderError) as exc_info:
        embedding.embed(["hello"])

    message = str(exc_info.value)
    assert "ollama request failed" in message
    assert "secret-key" not in message
