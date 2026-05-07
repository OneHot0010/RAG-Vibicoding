"""Smoke tests for Ollama LLM provider with mocked HTTP."""

from __future__ import annotations

from typing import Any
from urllib.error import URLError

import pytest

from core.settings import LLMSettings
from libs.llm import ChatMessage, LLMFactory, LLMProviderError, OllamaLLM


@pytest.fixture(autouse=True)
def register_ollama_provider() -> None:
    LLMFactory.clear()
    LLMFactory.register("ollama", OllamaLLM)


def test_factory_creates_ollama_provider() -> None:
    llm = LLMFactory.create(LLMSettings(provider="ollama", model="llama3.1"))

    assert isinstance(llm, OllamaLLM)


def test_ollama_chat_posts_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(self: OllamaLLM, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured.update(url=url, payload=payload)
        return {"message": {"content": "local answer"}}

    monkeypatch.setattr(OllamaLLM, "_post_json", fake_post)
    llm = OllamaLLM(
        LLMSettings(provider="ollama", model="llama3.1", base_url="http://ollama.test:11434")
    )

    response = llm.chat([ChatMessage(role="user", content="Hi")])

    assert response == "local answer"
    assert captured["url"] == "http://ollama.test:11434/api/chat"
    assert captured["payload"] == {
        "model": "llama3.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    }


def test_ollama_uses_localhost_default_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(self: OllamaLLM, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured["url"] = url
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(OllamaLLM, "_post_json", fake_post)
    llm = OllamaLLM(LLMSettings(provider="ollama", model="llama3.1"))

    assert llm.chat([{"role": "user", "content": "Hi"}]) == "ok"
    assert captured["url"] == "http://localhost:11434/api/chat"


def test_message_shape_error_mentions_ollama_and_field() -> None:
    llm = OllamaLLM(LLMSettings(provider="ollama", model="llama3.1"))

    with pytest.raises(LLMProviderError, match=r"ollama message\[0\]\.content"):
        llm.chat([{"role": "user"}])


def test_response_shape_error_mentions_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(self: OllamaLLM, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"done": True}

    monkeypatch.setattr(OllamaLLM, "_post_json", fake_post)
    llm = OllamaLLM(LLMSettings(provider="ollama", model="llama3.1"))

    with pytest.raises(LLMProviderError, match="ollama response missing message.content"):
        llm.chat([{"role": "user", "content": "Hi"}])


def test_connection_error_is_readable_and_does_not_leak_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*args: Any, **kwargs: Any) -> Any:
        raise URLError("secret-token should not be exposed")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm = OllamaLLM(
        LLMSettings(
            provider="ollama",
            model="llama3.1",
            base_url="http://localhost:11434",
            api_key="secret-token",
        )
    )

    with pytest.raises(LLMProviderError) as exc_info:
        llm.chat([{"role": "user", "content": "Hi"}])

    message = str(exc_info.value)
    assert "ollama request failed" in message
    assert "secret-token" not in message
