"""Smoke tests for OpenAI-compatible LLM providers with mocked HTTP."""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import LLMSettings
from libs.llm import ArkLLM, AzureLLM, ChatMessage, DeepSeekLLM, LLMFactory, LLMProviderError, OpenAILLM


@pytest.fixture(autouse=True)
def register_default_providers() -> None:
    LLMFactory.clear()
    LLMFactory.register("openai", OpenAILLM)
    LLMFactory.register("azure", AzureLLM)
    LLMFactory.register("ark", ArkLLM)
    LLMFactory.register("volcengine", ArkLLM)
    LLMFactory.register("deepseek", DeepSeekLLM)


def test_openai_llm_chat_posts_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(self: OpenAILLM, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        captured.update(url=url, payload=payload, headers=headers)
        return {"choices": [{"message": {"content": "hello"}}]}

    monkeypatch.setattr(OpenAILLM, "_post_json", fake_post)
    llm = LLMFactory.create(
        LLMSettings(provider="openai", model="gpt-4o-mini", api_key="secret")
    )

    response = llm.chat([ChatMessage(role="user", content="Hi")])

    assert response == "hello"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["payload"] == {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    assert captured["headers"]["Authorization"] == "Bearer secret"


def test_deepseek_llm_uses_deepseek_default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(self: DeepSeekLLM, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        captured.update(url=url, payload=payload, headers=headers)
        return {"choices": [{"message": {"content": "deep answer"}}]}

    monkeypatch.setattr(DeepSeekLLM, "_post_json", fake_post)
    llm = LLMFactory.create(
        LLMSettings(provider="deepseek", model="deepseek-chat", api_key="deep-secret")
    )

    response = llm.chat([{"role": "user", "content": "Hi"}])

    assert response == "deep answer"
    assert captured["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert captured["payload"]["model"] == "deepseek-chat"
    assert captured["headers"]["Authorization"] == "Bearer deep-secret"


def test_azure_llm_uses_deployment_url_and_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(self: AzureLLM, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        captured.update(url=url, payload=payload, headers=headers)
        return {"choices": [{"message": {"content": "azure answer"}}]}

    monkeypatch.setattr(AzureLLM, "_post_json", fake_post)
    llm = LLMFactory.create(
        LLMSettings(
            provider="azure",
            model="gpt-4o",
            deployment="prod-gpt-4o",
            azure_endpoint="https://example.openai.azure.com",
            api_key="azure-secret",
            api_version="2024-10-21",
        )
    )

    response = llm.chat([{"role": "user", "content": "Hi"}])

    assert response == "azure answer"
    assert captured["url"] == (
        "https://example.openai.azure.com/openai/deployments/prod-gpt-4o/"
        "chat/completions?api-version=2024-10-21"
    )
    assert "model" not in captured["payload"]
    assert captured["headers"]["api-key"] == "azure-secret"


def test_azure_llm_requires_endpoint() -> None:
    llm = AzureLLM(LLMSettings(provider="azure", model="gpt-4o"))

    with pytest.raises(LLMProviderError, match="azure requires"):
        llm.chat([{"role": "user", "content": "Hi"}])


def test_ark_llm_uses_openai_compatible_ark_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(self: ArkLLM, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        captured.update(url=url, payload=payload, headers=headers)
        return {"choices": [{"message": {"content": "ark answer"}}]}

    monkeypatch.setattr(ArkLLM, "_post_json", fake_post)
    llm = LLMFactory.create(
        LLMSettings(provider="ark", model="ep-20260509-demo", api_key="ark-secret")
    )

    response = llm.chat([{"role": "user", "content": "Hi"}])

    assert response == "ark answer"
    assert captured["url"] == "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    assert captured["payload"] == {
        "model": "ep-20260509-demo",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    assert captured["headers"]["Authorization"] == "Bearer ark-secret"


def test_ark_llm_accepts_custom_base_url_and_volcengine_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(self: ArkLLM, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        captured["url"] = url
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(ArkLLM, "_post_json", fake_post)
    llm = LLMFactory.create(
        LLMSettings(
            provider="volcengine",
            model="ep-custom",
            base_url="https://ark-custom.example.com/api/v3",
        )
    )

    assert llm.chat([{"role": "user", "content": "Hi"}]) == "ok"
    assert captured["url"] == "https://ark-custom.example.com/api/v3/chat/completions"


def test_message_shape_error_mentions_provider_and_field() -> None:
    llm = OpenAILLM(LLMSettings(provider="openai", model="gpt-4o-mini"))

    with pytest.raises(LLMProviderError, match=r"openai message\[0\]\.content"):
        llm.chat([{"role": "user"}])


def test_response_shape_error_mentions_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(self: OpenAILLM, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        return {"choices": []}

    monkeypatch.setattr(OpenAILLM, "_post_json", fake_post)
    llm = OpenAILLM(LLMSettings(provider="openai", model="gpt-4o-mini"))

    with pytest.raises(LLMProviderError, match="openai response missing"):
        llm.chat([{"role": "user", "content": "Hi"}])


def test_factory_routes_builtin_provider_names() -> None:
    assert isinstance(LLMFactory.create(LLMSettings(provider="openai", model="gpt-4o-mini")), OpenAILLM)
    assert isinstance(
        LLMFactory.create(
            LLMSettings(
                provider="azure",
                model="gpt-4o",
                azure_endpoint="https://example.openai.azure.com",
            )
        ),
        AzureLLM,
    )
    assert isinstance(LLMFactory.create(LLMSettings(provider="deepseek", model="deepseek-chat")), DeepSeekLLM)
    assert isinstance(LLMFactory.create(LLMSettings(provider="ark", model="ep-demo")), ArkLLM)
    assert isinstance(LLMFactory.create(LLMSettings(provider="volcengine", model="ep-demo")), ArkLLM)
