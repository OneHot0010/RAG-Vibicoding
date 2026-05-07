"""OpenAI-compatible embedding providers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from core.settings import EmbeddingSettings
from libs.embedding.base_embedding import BaseEmbedding


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding provider request or response is invalid."""


class OpenAIEmbedding(BaseEmbedding):
    """Minimal OpenAI embeddings client."""

    provider_name = "openai"
    default_base_url = "https://api.openai.com/v1"

    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        normalized_texts = _normalize_texts(texts, self.provider_name)
        payload = {"model": self.settings.model, "input": normalized_texts}
        response = self._post_json(self._embeddings_url(), payload, self._headers())
        return _extract_embedding_response(response, len(normalized_texts), self.provider_name)

    def _embeddings_url(self) -> str:
        return f"{self._base_url().rstrip('/')}/embeddings"

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
            raise EmbeddingProviderError(f"{self.provider_name} request failed: {exc.__class__.__name__}") from exc

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EmbeddingProviderError(f"{self.provider_name} returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise EmbeddingProviderError(f"{self.provider_name} response must be a JSON object")
        return decoded


def _normalize_texts(texts: list[str], provider_name: str) -> list[str]:
    if not isinstance(texts, list):
        raise EmbeddingProviderError(f"{provider_name} texts must be a list of strings")
    if not texts:
        raise EmbeddingProviderError(f"{provider_name} texts must not be empty")
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text:
            raise EmbeddingProviderError(f"{provider_name} texts[{index}] must be a non-empty string")
    return texts


def _extract_embedding_response(
    response: dict[str, Any],
    expected_count: int,
    provider_name: str,
) -> list[list[float]]:
    data = response.get("data")
    if not isinstance(data, list):
        raise EmbeddingProviderError(f"{provider_name} response missing data list")
    if len(data) != expected_count:
        raise EmbeddingProviderError(
            f"{provider_name} response embedding count mismatch: expected {expected_count}, got {len(data)}"
        )

    vectors: list[list[float]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict) or "embedding" not in item:
            raise EmbeddingProviderError(f"{provider_name} response data[{index}].embedding is required")
        embedding = item["embedding"]
        if not isinstance(embedding, list) or not all(isinstance(value, (int, float)) for value in embedding):
            raise EmbeddingProviderError(f"{provider_name} response data[{index}].embedding must be numeric")
        vectors.append([float(value) for value in embedding])
    return vectors
