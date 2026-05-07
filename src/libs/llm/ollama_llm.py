"""Ollama local chat provider."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping, Sequence

from core.settings import LLMSettings
from libs.llm.base_llm import BaseLLM, ChatMessage
from libs.llm.openai_llm import LLMProviderError, _normalize_messages


class OllamaLLM(BaseLLM):
    """Minimal Ollama `/api/chat` client."""

    provider_name = "ollama"
    default_base_url = "http://localhost:11434"

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    def chat(self, messages: Sequence[ChatMessage | Mapping[str, Any]]) -> str:
        normalized_messages = _normalize_messages(messages, self.provider_name)
        payload = {
            "model": self.settings.model,
            "messages": normalized_messages,
            "stream": False,
        }
        response = self._post_json(self._chat_url(), payload)
        return _extract_ollama_response(response)

    def _chat_url(self) -> str:
        return f"{self._base_url().rstrip('/')}/api/chat"

    def _base_url(self) -> str:
        return self.settings.base_url or self.default_base_url

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except (TimeoutError, urllib.error.URLError) as exc:
            raise LLMProviderError(f"{self.provider_name} request failed: {exc.__class__.__name__}") from exc

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMProviderError(f"{self.provider_name} returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise LLMProviderError(f"{self.provider_name} response must be a JSON object")
        return decoded


def _extract_ollama_response(response: dict[str, Any]) -> str:
    try:
        content = response["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise LLMProviderError("ollama response missing message.content") from exc
    if not isinstance(content, str):
        raise LLMProviderError("ollama response content must be a string")
    return content
