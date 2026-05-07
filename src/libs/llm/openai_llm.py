"""OpenAI-compatible chat completion providers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping, Sequence

from core.settings import LLMSettings
from libs.llm.base_llm import BaseLLM, ChatMessage


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider request or response is invalid."""


class OpenAICompatibleLLM(BaseLLM):
    """Minimal OpenAI-compatible chat completions client."""

    provider_name = "openai"
    default_base_url = "https://api.openai.com/v1"

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    def chat(self, messages: Sequence[ChatMessage | Mapping[str, Any]]) -> str:
        normalized_messages = _normalize_messages(messages, self.provider_name)
        payload = {"model": self.settings.model, "messages": normalized_messages}
        response = self._post_json(self._chat_completions_url(), payload, self._headers())
        return _extract_text_response(response, self.provider_name)

    def _chat_completions_url(self) -> str:
        return f"{self._base_url().rstrip('/')}/chat/completions"

    def _base_url(self) -> str:
        return self.settings.base_url or self.default_base_url

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def _post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise LLMProviderError(f"{self.provider_name} request failed: {exc}") from exc

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMProviderError(f"{self.provider_name} returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise LLMProviderError(f"{self.provider_name} response must be a JSON object")
        return decoded


class OpenAILLM(OpenAICompatibleLLM):
    """OpenAI official API chat completion client."""

    provider_name = "openai"
    default_base_url = "https://api.openai.com/v1"


def _normalize_messages(
    messages: Sequence[ChatMessage | Mapping[str, Any]],
    provider_name: str,
) -> list[dict[str, str]]:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise LLMProviderError(f"{provider_name} messages must be a sequence")

    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if isinstance(message, ChatMessage):
            role = message.role
            content = message.content
        elif isinstance(message, Mapping):
            role = message.get("role")
            content = message.get("content")
        else:
            raise LLMProviderError(f"{provider_name} message[{index}] must be a mapping or ChatMessage")

        if not isinstance(role, str) or not role:
            raise LLMProviderError(f"{provider_name} message[{index}].role is required")
        if not isinstance(content, str) or not content:
            raise LLMProviderError(f"{provider_name} message[{index}].content is required")
        normalized.append({"role": role, "content": content})
    return normalized


def _extract_text_response(response: dict[str, Any], provider_name: str) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMProviderError(f"{provider_name} response missing choices[0].message.content") from exc
    if not isinstance(content, str):
        raise LLMProviderError(f"{provider_name} response content must be a string")
    return content
