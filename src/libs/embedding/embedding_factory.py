"""Factory for creating configured embedding providers."""

from __future__ import annotations

from typing import Callable, ClassVar

from core.settings import EmbeddingSettings, Settings
from libs.embedding.base_embedding import BaseEmbedding


EmbeddingProviderBuilder = Callable[[EmbeddingSettings], BaseEmbedding]


class EmbeddingFactoryError(ValueError):
    """Raised when an embedding provider cannot be created from settings."""


class EmbeddingFactory:
    """Registry-backed factory for embedding providers."""

    _providers: ClassVar[dict[str, EmbeddingProviderBuilder]] = {}

    @classmethod
    def register(cls, provider: str, builder: EmbeddingProviderBuilder) -> None:
        """Register a provider builder by name."""
        normalized = cls._normalize_provider(provider)
        if not callable(builder):
            raise TypeError("Embedding provider builder must be callable")
        cls._providers[normalized] = builder

    @classmethod
    def unregister(cls, provider: str) -> None:
        """Remove a provider registration if it exists."""
        cls._providers.pop(cls._normalize_provider(provider), None)

    @classmethod
    def clear(cls) -> None:
        """Clear provider registrations for isolated tests."""
        cls._providers.clear()

    @classmethod
    def create(cls, settings: Settings | EmbeddingSettings) -> BaseEmbedding:
        """Create an embedding backend from full or embedding-specific settings."""
        embedding_settings = settings.embedding if isinstance(settings, Settings) else settings
        provider = cls._normalize_provider(embedding_settings.provider)

        try:
            builder = cls._providers[provider]
        except KeyError as exc:
            available = ", ".join(sorted(cls._providers)) or "none"
            raise EmbeddingFactoryError(
                f"Unknown embedding provider: {embedding_settings.provider}. "
                f"Registered providers: {available}"
            ) from exc

        embedding = builder(embedding_settings)
        if not isinstance(embedding, BaseEmbedding):
            raise EmbeddingFactoryError(
                f"Embedding provider '{provider}' returned {type(embedding).__name__}, "
                "expected BaseEmbedding"
            )
        return embedding

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        if not provider or not provider.strip():
            raise EmbeddingFactoryError("Embedding provider name is required")
        return provider.strip().lower()
