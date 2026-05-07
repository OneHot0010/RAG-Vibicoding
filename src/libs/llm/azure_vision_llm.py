"""Azure OpenAI vision chat completion provider."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import quote

from core.settings import VisionLLMSettings
from libs.llm.base_vision_llm import BaseVisionLLM, ChatResponse
from libs.llm.openai_llm import LLMProviderError, OpenAICompatibleLLM, _extract_text_response


class AzureVisionLLM(BaseVisionLLM):
    """Azure OpenAI vision client for text plus single-image prompts."""

    provider_name = "azure-vision"
    default_api_version = "2024-02-15-preview"

    def __init__(self, settings: VisionLLMSettings) -> None:
        self.settings = settings

    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any | None = None,
    ) -> ChatResponse:
        if not isinstance(text, str) or not text.strip():
            raise LLMProviderError("azure vision prompt text must be a non-empty string")

        image_url, image_metadata = _image_to_data_url(image_path)
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ]
        }
        response = self._post_json(self._chat_completions_url(), payload, self._headers())
        return ChatResponse(
            content=_extract_text_response(response, self.provider_name),
            metadata={
                "provider": "azure",
                "model": self.settings.model,
                **image_metadata,
            },
        )

    def _chat_completions_url(self) -> str:
        endpoint = self.settings.azure_endpoint or self.settings.base_url
        if not endpoint:
            raise LLMProviderError("azure vision requires vision_llm.azure_endpoint or vision_llm.base_url")
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

    def _post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        return OpenAICompatibleLLM._post_json(self, url, payload, headers)


def _image_to_data_url(image_path: str | bytes) -> tuple[str, dict[str, Any]]:
    if isinstance(image_path, bytes):
        if not image_path:
            raise LLMProviderError("azure vision image bytes must not be empty")
        mime_type = "image/png"
        image_bytes = image_path
        source = "bytes"
    elif isinstance(image_path, str):
        path = Path(image_path)
        if not path.exists():
            raise LLMProviderError(f"azure vision image file not found: {path}")
        image_bytes = path.read_bytes()
        if not image_bytes:
            raise LLMProviderError(f"azure vision image file is empty: {path}")
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        source = str(path)
    else:
        raise LLMProviderError("azure vision image_path must be a path string or bytes")

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return (
        f"data:{mime_type};base64,{encoded}",
        {
            "image_source": source,
            "image_mime_type": mime_type,
            "image_size": len(image_bytes),
        },
    )
