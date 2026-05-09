"""Tests for the Azure Vision LLM provider with mocked HTTP."""

from __future__ import annotations

import base64
from typing import Any

import pytest

from core.settings import VisionLLMSettings
from libs.llm import ArkVisionLLM, AzureVisionLLM, ChatResponse, LLMFactory, LLMProviderError


@pytest.fixture(autouse=True)
def register_default_vision_provider() -> None:
    LLMFactory.clear()
    LLMFactory.register_vision("azure", AzureVisionLLM)
    LLMFactory.register_vision("ark", ArkVisionLLM)
    LLMFactory.register_vision("volcengine", ArkVisionLLM)


def settings() -> VisionLLMSettings:
    return VisionLLMSettings(
        provider="azure",
        model="gpt-4o",
        deployment="prod-gpt-4o",
        azure_endpoint="https://example.openai.azure.com",
        api_key="azure-secret",
        api_version="2024-10-21",
    )


def test_azure_vision_posts_expected_payload_for_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(
        self: AzureVisionLLM,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        captured.update(url=url, payload=payload, headers=headers)
        return {"choices": [{"message": {"content": "caption"}}]}

    monkeypatch.setattr(AzureVisionLLM, "_post_json", fake_post)
    vision_llm = LLMFactory.create_vision_llm(settings())

    response = vision_llm.chat_with_image("Describe this image.", b"image-bytes")

    assert response == ChatResponse(
        content="caption",
        metadata={
            "provider": "azure",
            "model": "gpt-4o",
            "image_source": "bytes",
            "image_mime_type": "image/png",
            "image_size": 11,
        },
    )
    assert captured["url"] == (
        "https://example.openai.azure.com/openai/deployments/prod-gpt-4o/"
        "chat/completions?api-version=2024-10-21"
    )
    assert captured["headers"]["api-key"] == "azure-secret"
    content = captured["payload"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Describe this image."}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == (
        "data:image/png;base64," + base64.b64encode(b"image-bytes").decode("ascii")
    )


def test_azure_vision_reads_image_file_and_detects_mime_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured: dict[str, Any] = {}
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"jpg-bytes")

    def fake_post(
        self: AzureVisionLLM,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        captured.update(payload=payload)
        return {"choices": [{"message": {"content": "file caption"}}]}

    monkeypatch.setattr(AzureVisionLLM, "_post_json", fake_post)
    vision_llm = AzureVisionLLM(settings())

    response = vision_llm.chat_with_image("Describe file.", str(image_path))

    assert response.content == "file caption"
    assert response.metadata["image_source"] == str(image_path)
    assert response.metadata["image_mime_type"] == "image/jpeg"
    assert captured["payload"]["messages"][0]["content"][1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )


def test_azure_vision_uses_model_as_deployment_when_deployment_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(
        self: AzureVisionLLM,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        captured["url"] = url
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(AzureVisionLLM, "_post_json", fake_post)
    vision_llm = AzureVisionLLM(
        VisionLLMSettings(
            provider="azure",
            model="gpt-4o-mini",
            azure_endpoint="https://example.openai.azure.com",
        )
    )

    vision_llm.chat_with_image("Describe.", b"x")

    assert captured["url"] == (
        "https://example.openai.azure.com/openai/deployments/gpt-4o-mini/"
        "chat/completions?api-version=2024-02-15-preview"
    )


def test_factory_can_create_registered_azure_vision_llm() -> None:
    vision_llm = LLMFactory.create_vision_llm(settings())

    assert isinstance(vision_llm, AzureVisionLLM)


def test_ark_vision_posts_openai_compatible_payload_for_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(
        self: ArkVisionLLM,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        captured.update(url=url, payload=payload, headers=headers)
        return {"choices": [{"message": {"content": "ark caption"}}]}

    monkeypatch.setattr(ArkVisionLLM, "_post_json", fake_post)
    vision_llm = LLMFactory.create_vision_llm(
        VisionLLMSettings(provider="ark", model="ep-vision", api_key="ark-secret")
    )

    response = vision_llm.chat_with_image("Describe this image.", b"image-bytes")

    assert response == ChatResponse(
        content="ark caption",
        metadata={
            "provider": "ark",
            "model": "ep-vision",
            "image_source": "bytes",
            "image_mime_type": "image/png",
            "image_size": 11,
        },
    )
    assert captured["url"] == "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer ark-secret"
    assert captured["payload"]["model"] == "ep-vision"
    content = captured["payload"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Describe this image."}
    assert content[1]["image_url"]["url"] == (
        "data:image/png;base64," + base64.b64encode(b"image-bytes").decode("ascii")
    )


def test_ark_vision_accepts_volcengine_alias() -> None:
    vision_llm = LLMFactory.create_vision_llm(VisionLLMSettings(provider="volcengine", model="ep-vision"))

    assert isinstance(vision_llm, ArkVisionLLM)


def test_azure_vision_requires_endpoint() -> None:
    vision_llm = AzureVisionLLM(VisionLLMSettings(provider="azure", model="gpt-4o"))

    with pytest.raises(LLMProviderError, match="azure vision requires"):
        vision_llm.chat_with_image("Describe.", b"x")


def test_azure_vision_requires_non_empty_prompt() -> None:
    vision_llm = AzureVisionLLM(settings())

    with pytest.raises(LLMProviderError, match="non-empty"):
        vision_llm.chat_with_image(" ", b"x")


def test_azure_vision_rejects_missing_image_file() -> None:
    vision_llm = AzureVisionLLM(settings())

    with pytest.raises(LLMProviderError, match="image file not found"):
        vision_llm.chat_with_image("Describe.", "missing.png")


def test_azure_vision_rejects_empty_image_bytes() -> None:
    vision_llm = AzureVisionLLM(settings())

    with pytest.raises(LLMProviderError, match="bytes must not be empty"):
        vision_llm.chat_with_image("Describe.", b"")


def test_azure_vision_response_shape_error_mentions_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(
        self: AzureVisionLLM,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        return {"choices": []}

    monkeypatch.setattr(AzureVisionLLM, "_post_json", fake_post)
    vision_llm = AzureVisionLLM(settings())

    with pytest.raises(LLMProviderError, match="azure-vision response missing"):
        vision_llm.chat_with_image("Describe.", b"x")
