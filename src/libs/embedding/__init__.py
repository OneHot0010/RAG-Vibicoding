"""Embedding abstractions, providers, and factory helpers."""

from libs.embedding.ark_embedding import ArkEmbedding
from libs.embedding.azure_embedding import AzureEmbedding
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory, EmbeddingFactoryError
from libs.embedding.ollama_embedding import OllamaEmbedding
from libs.embedding.openai_embedding import EmbeddingProviderError, OpenAIEmbedding

EmbeddingFactory.register("openai", OpenAIEmbedding)
EmbeddingFactory.register("azure", AzureEmbedding)
EmbeddingFactory.register("ark", ArkEmbedding)
EmbeddingFactory.register("volcengine", ArkEmbedding)
EmbeddingFactory.register("volcano", ArkEmbedding)
EmbeddingFactory.register("ollama", OllamaEmbedding)

__all__ = [
    "ArkEmbedding",
    "AzureEmbedding",
    "BaseEmbedding",
    "EmbeddingFactory",
    "EmbeddingFactoryError",
    "EmbeddingProviderError",
    "OllamaEmbedding",
    "OpenAIEmbedding",
]
