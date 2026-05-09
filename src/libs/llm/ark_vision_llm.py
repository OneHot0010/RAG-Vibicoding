"""Volcengine Ark OpenAI-compatible vision chat provider."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from core.settings import VisionLLMSettings
from libs.llm.base_vision_llm import BaseVisionLLM, ChatResponse
from libs.llm.openai_llm import LLMProviderError, OpenAICompatibleLLM, _extract_text_response


class ArkVisionLLM(BaseVisionLLM):
    """Volcengine Ark vision client for text plus single-image prompts."""

    provider_name = "ark-vision"
    default_base_url = "https://ark.cn-beijing.volces.com/api/v3"

    def __init__(self, settings: VisionLLMSettings) -> None:
        self.settings = settings

    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any | None = None,
    ) -> ChatResponse:
        if not isinstance(text, str) or not text.strip():
            raise LLMProviderError("ark vision prompt text must be a non-empty string")

        image_url, image_metadata = _image_to_data_url(image_path)
        payload = {
            "model": self.settings.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        }
        response = self._post_json(self._chat_completions_url(), payload, self._headers())
        return ChatResponse(
            content=_extract_text_response(response, self.provider_name),
            metadata={
                "provider": "ark",
                "model": self.settings.model,
                **image_metadata,
            },
        )

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
        return OpenAICompatibleLLM._post_json(self, url, payload, headers)


def _image_to_data_url(image_path: str | bytes) -> tuple[str, dict[str, Any]]:
    if isinstance(image_path, bytes):
        if not image_path:
            raise LLMProviderError("ark vision image bytes must not be empty")
        mime_type = "image/png"
        image_bytes = image_path
        source = "bytes"
    elif isinstance(image_path, str):
        path = Path(image_path)
        if not path.exists():
            raise LLMProviderError(f"ark vision image file not found: {path}")
        image_bytes = path.read_bytes()
        if not image_bytes:
            raise LLMProviderError(f"ark vision image file is empty: {path}")
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        source = str(path)
    else:
        raise LLMProviderError("ark vision image_path must be a path string or bytes")

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return (
        f"data:{mime_type};base64,{encoded}",
        {
            "image_source": source,
            "image_mime_type": mime_type,
            "image_size": len(image_bytes),
        },
    )
