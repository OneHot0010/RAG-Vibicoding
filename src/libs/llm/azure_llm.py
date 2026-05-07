"""Azure OpenAI chat completion provider."""

from __future__ import annotations

from typing import Any, Mapping, Sequence
from urllib.parse import quote

from core.settings import LLMSettings
from libs.llm.base_llm import ChatMessage
from libs.llm.openai_llm import OpenAICompatibleLLM, _extract_text_response, _normalize_messages


class AzureLLM(OpenAICompatibleLLM):
    """Azure OpenAI chat completion client."""

    provider_name = "azure"
    default_api_version = "2024-02-15-preview"

    def chat(self, messages: Sequence[ChatMessage | Mapping[str, Any]]) -> str:
        normalized_messages = _normalize_messages(messages, self.provider_name)
        payload = {"messages": normalized_messages}
        response = self._post_json(self._chat_completions_url(), payload, self._headers())
        return _extract_text_response(response, self.provider_name)

    def _chat_completions_url(self) -> str:
        endpoint = self.settings.azure_endpoint or self.settings.base_url
        if not endpoint:
            from libs.llm.openai_llm import LLMProviderError

            raise LLMProviderError("azure requires llm.azure_endpoint or llm.base_url")
        deployment = self.settings.deployment or self.settings.model
        api_version = self.settings.api_version or self.default_api_version
        return (
            f"{endpoint.rstrip('/')}/openai/deployments/{quote(deployment)}/chat/completions"
            f"?api-version={quote(api_version)}"
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["api-key"] = self.settings.api_key
        return headers
