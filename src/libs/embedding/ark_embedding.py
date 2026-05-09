"""Volcengine Ark OpenAI-compatible embedding provider."""

from __future__ import annotations

from libs.embedding.openai_embedding import OpenAIEmbedding


class ArkEmbedding(OpenAIEmbedding):
    """Volcengine Ark embeddings client using the OpenAI-compatible API."""

    provider_name = "ark"
    default_base_url = "https://ark.cn-beijing.volces.com/api/v3"
