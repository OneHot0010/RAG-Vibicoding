"""Azure OpenAI embedding provider."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from libs.embedding.openai_embedding import (
    EmbeddingProviderError,
    OpenAIEmbedding,
    _extract_embedding_response,
    _normalize_texts,
)


class AzureEmbedding(OpenAIEmbedding):
    """Azure OpenAI embeddings client."""

    provider_name = "azure"
    default_api_version = "2024-02-15-preview"

    def embed(self, texts: list[str], trace: Any | None = None) -> list[list[float]]:
        normalized_texts = _normalize_texts(texts, self.provider_name)
        payload = {"input": normalized_texts}
        response = self._post_json(self._embeddings_url(), payload, self._headers())
        return _extract_embedding_response(response, len(normalized_texts), self.provider_name)

    def _embeddings_url(self) -> str:
        endpoint = self.settings.azure_endpoint or self.settings.base_url
        if not endpoint:
            raise EmbeddingProviderError("azure embedding requires embedding.azure_endpoint or embedding.base_url")
        deployment = self.settings.deployment or self.settings.model
        api_version = self.settings.api_version or self.default_api_version
        return (
            f"{endpoint.rstrip('/')}/openai/deployments/{quote(deployment)}/embeddings"
            f"?api-version={quote(api_version)}"
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["api-key"] = self.settings.api_key
        return headers
