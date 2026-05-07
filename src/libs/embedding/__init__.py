"""Embedding abstractions, providers, and factory helpers."""

from libs.embedding.azure_embedding import AzureEmbedding
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory, EmbeddingFactoryError
from libs.embedding.openai_embedding import EmbeddingProviderError, OpenAIEmbedding

EmbeddingFactory.register("openai", OpenAIEmbedding)
EmbeddingFactory.register("azure", AzureEmbedding)

__all__ = [
    "AzureEmbedding",
    "BaseEmbedding",
    "EmbeddingFactory",
    "EmbeddingFactoryError",
    "EmbeddingProviderError",
    "OpenAIEmbedding",
]
