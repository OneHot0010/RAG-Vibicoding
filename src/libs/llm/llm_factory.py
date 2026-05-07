"""Factory for creating configured LLM providers."""

from __future__ import annotations

from typing import Callable, ClassVar

from core.settings import LLMSettings, Settings
from libs.llm.base_llm import BaseLLM


LLMProviderBuilder = Callable[[LLMSettings], BaseLLM]


class LLMFactoryError(ValueError):
    """Raised when an LLM provider cannot be created from settings."""


class LLMFactory:
    """Registry-backed factory for LLM providers."""

    _providers: ClassVar[dict[str, LLMProviderBuilder]] = {}

    @classmethod
    def register(cls, provider: str, builder: LLMProviderBuilder) -> None:
        """Register a provider builder by name."""
        normalized = cls._normalize_provider(provider)
        if not callable(builder):
            raise TypeError("LLM provider builder must be callable")
        cls._providers[normalized] = builder

    @classmethod
    def unregister(cls, provider: str) -> None:
        """Remove a provider registration if it exists."""
        cls._providers.pop(cls._normalize_provider(provider), None)

    @classmethod
    def clear(cls) -> None:
        """Clear all provider registrations.

        This is primarily useful in unit tests where fake providers should not
        leak between test cases.
        """
        cls._providers.clear()

    @classmethod
    def create(cls, settings: Settings | LLMSettings) -> BaseLLM:
        """Create an LLM from full project settings or direct LLM settings."""
        llm_settings = settings.llm if isinstance(settings, Settings) else settings
        provider = cls._normalize_provider(llm_settings.provider)

        try:
            builder = cls._providers[provider]
        except KeyError as exc:
            available = ", ".join(sorted(cls._providers)) or "none"
            raise LLMFactoryError(
                f"Unknown LLM provider: {llm_settings.provider}. "
                f"Registered providers: {available}"
            ) from exc

        llm = builder(llm_settings)
        if not isinstance(llm, BaseLLM):
            raise LLMFactoryError(
                f"LLM provider '{provider}' returned {type(llm).__name__}, "
                "expected BaseLLM"
            )
        return llm

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        if not provider or not provider.strip():
            raise LLMFactoryError("LLM provider name is required")
        return provider.strip().lower()
