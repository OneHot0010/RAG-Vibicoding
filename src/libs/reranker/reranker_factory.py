"""Factory for creating configured rerankers."""

from __future__ import annotations

from typing import Callable, ClassVar

from core.settings import RerankSettings, Settings
from libs.reranker.base_reranker import BaseReranker, NoneReranker


RerankerBuilder = Callable[[RerankSettings], BaseReranker]


class RerankerFactoryError(ValueError):
    """Raised when a reranker backend cannot be created from settings."""


class RerankerFactory:
    """Registry-backed factory for reranker backends."""

    _backends: ClassVar[dict[str, RerankerBuilder]] = {"none": lambda settings: NoneReranker()}

    @classmethod
    def register(cls, backend: str, builder: RerankerBuilder) -> None:
        """Register a reranker builder by backend name."""
        normalized = cls._normalize_backend(backend)
        if not callable(builder):
            raise TypeError("Reranker builder must be callable")
        cls._backends[normalized] = builder

    @classmethod
    def unregister(cls, backend: str) -> None:
        """Remove a backend registration if it exists."""
        normalized = cls._normalize_backend(backend)
        if normalized == "none":
            cls._backends[normalized] = lambda settings: NoneReranker()
            return
        cls._backends.pop(normalized, None)

    @classmethod
    def reset_defaults(cls) -> None:
        """Reset registrations to built-in defaults."""
        cls._backends = {"none": lambda settings: NoneReranker()}

    @classmethod
    def create(cls, settings: Settings | RerankSettings) -> BaseReranker:
        """Create a reranker from full project settings or rerank settings."""
        rerank_settings = settings.rerank if isinstance(settings, Settings) else settings
        backend = cls._normalize_backend(rerank_settings.backend)

        try:
            builder = cls._backends[backend]
        except KeyError as exc:
            available = ", ".join(sorted(cls._backends)) or "none"
            raise RerankerFactoryError(
                f"Unknown reranker backend: {rerank_settings.backend}. "
                f"Registered backends: {available}"
            ) from exc

        reranker = builder(rerank_settings)
        if not isinstance(reranker, BaseReranker):
            raise RerankerFactoryError(
                f"Reranker backend '{backend}' returned {type(reranker).__name__}, "
                "expected BaseReranker"
            )
        return reranker

    @staticmethod
    def _normalize_backend(backend: str) -> str:
        if not backend or not backend.strip():
            raise RerankerFactoryError("Reranker backend name is required")
        return backend.strip().lower()
