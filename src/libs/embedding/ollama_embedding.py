"""Ollama local embedding provider."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from core.settings import EmbeddingSettings
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.openai_embedding import EmbeddingProviderError, _normalize_texts


class OllamaEmbedding(BaseEmbedding):
    """Minimal Ollama `/api/embeddings` client."""

    provider_name = "ollama"
    default_base_url = "http://localhost:11434"

    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        normalized_texts = _normalize_texts(texts, self.provider_name)
        return [self._embed_one(text) for text in normalized_texts]

    def _embed_one(self, text: str) -> list[float]:
        payload = {"model": self.settings.model, "prompt": text}
        response = self._post_json(self._embeddings_url(), payload)
        return _extract_ollama_embedding(response)

    def _embeddings_url(self) -> str:
        return f"{self._base_url().rstrip('/')}/api/embeddings"

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
            raise EmbeddingProviderError(f"{self.provider_name} request failed: {exc.__class__.__name__}") from exc

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EmbeddingProviderError(f"{self.provider_name} returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise EmbeddingProviderError(f"{self.provider_name} response must be a JSON object")
        return decoded


def _extract_ollama_embedding(response: dict[str, Any]) -> list[float]:
    embedding = response.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise EmbeddingProviderError("ollama response missing embedding list")
    if not all(isinstance(value, (int, float)) for value in embedding):
        raise EmbeddingProviderError("ollama response embedding must be numeric")
    return [float(value) for value in embedding]
