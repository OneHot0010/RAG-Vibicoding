"""Factory for creating configured text splitters."""

from __future__ import annotations

from typing import Callable, ClassVar

from core.settings import Settings, SplitterSettings
from libs.splitter.base_splitter import BaseSplitter


SplitterBuilder = Callable[[SplitterSettings], BaseSplitter]


class SplitterFactoryError(ValueError):
    """Raised when a splitter strategy cannot be created from settings."""


class SplitterFactory:
    """Registry-backed factory for splitter strategies."""

    _strategies: ClassVar[dict[str, SplitterBuilder]] = {}

    @classmethod
    def register(cls, strategy: str, builder: SplitterBuilder) -> None:
        """Register a splitter builder by strategy name."""
        normalized = cls._normalize_strategy(strategy)
        if not callable(builder):
            raise TypeError("Splitter builder must be callable")
        cls._strategies[normalized] = builder

    @classmethod
    def unregister(cls, strategy: str) -> None:
        """Remove a strategy registration if it exists."""
        cls._strategies.pop(cls._normalize_strategy(strategy), None)

    @classmethod
    def clear(cls) -> None:
        """Clear strategy registrations for isolated tests."""
        cls._strategies.clear()

    @classmethod
    def create(cls, settings: Settings | SplitterSettings) -> BaseSplitter:
        """Create a splitter from full project settings or splitter settings."""
        splitter_settings = settings.splitter if isinstance(settings, Settings) else settings
        strategy = cls._normalize_strategy(splitter_settings.strategy)

        try:
            builder = cls._strategies[strategy]
        except KeyError as exc:
            available = ", ".join(sorted(cls._strategies)) or "none"
            raise SplitterFactoryError(
                f"Unknown splitter strategy: {splitter_settings.strategy}. "
                f"Registered strategies: {available}"
            ) from exc

        splitter = builder(splitter_settings)
        if not isinstance(splitter, BaseSplitter):
            raise SplitterFactoryError(
                f"Splitter strategy '{strategy}' returned {type(splitter).__name__}, "
                "expected BaseSplitter"
            )
        return splitter

    @staticmethod
    def _normalize_strategy(strategy: str) -> str:
        if not strategy or not strategy.strip():
            raise SplitterFactoryError("Splitter strategy name is required")
        return strategy.strip().lower()
