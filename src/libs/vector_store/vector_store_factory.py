"""Factory for creating configured vector stores."""

from __future__ import annotations

from typing import Callable, ClassVar

from core.settings import Settings, VectorStoreSettings
from libs.vector_store.base_vector_store import BaseVectorStore


VectorStoreBuilder = Callable[[VectorStoreSettings], BaseVectorStore]


class VectorStoreFactoryError(ValueError):
    """Raised when a vector store backend cannot be created from settings."""


class VectorStoreFactory:
    """Registry-backed factory for vector store backends."""

    _backends: ClassVar[dict[str, VectorStoreBuilder]] = {}

    @classmethod
    def register(cls, backend: str, builder: VectorStoreBuilder) -> None:
        """Register a vector store builder by backend name."""
        normalized = cls._normalize_backend(backend)
        if not callable(builder):
            raise TypeError("Vector store builder must be callable")
        cls._backends[normalized] = builder

    @classmethod
    def unregister(cls, backend: str) -> None:
        """Remove a backend registration if it exists."""
        cls._backends.pop(cls._normalize_backend(backend), None)

    @classmethod
    def clear(cls) -> None:
        """Clear backend registrations for isolated tests."""
        cls._backends.clear()

    @classmethod
    def create(cls, settings: Settings | VectorStoreSettings) -> BaseVectorStore:
        """Create a vector store from full project or vector-store settings."""
        vector_store_settings = settings.vector_store if isinstance(settings, Settings) else settings
        backend = cls._normalize_backend(vector_store_settings.backend)

        try:
            builder = cls._backends[backend]
        except KeyError as exc:
            available = ", ".join(sorted(cls._backends)) or "none"
            raise VectorStoreFactoryError(
                f"Unknown vector store backend: {vector_store_settings.backend}. "
                f"Registered backends: {available}"
            ) from exc

        vector_store = builder(vector_store_settings)
        if not isinstance(vector_store, BaseVectorStore):
            raise VectorStoreFactoryError(
                f"Vector store backend '{backend}' returned {type(vector_store).__name__}, "
                "expected BaseVectorStore"
            )
        return vector_store

    @staticmethod
    def _normalize_backend(backend: str) -> str:
        if not backend or not backend.strip():
            raise VectorStoreFactoryError("Vector store backend name is required")
        return backend.strip().lower()
