"""Factory for creating configured evaluators."""

from __future__ import annotations

from typing import Callable, ClassVar

from core.settings import EvaluationSettings, Settings
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator


EvaluatorBuilder = Callable[[EvaluationSettings], BaseEvaluator]


class EvaluatorFactoryError(ValueError):
    """Raised when an evaluator backend cannot be created from settings."""


class EvaluatorFactory:
    """Registry-backed factory for evaluator backends."""

    _backends: ClassVar[dict[str, EvaluatorBuilder]] = {
        "custom": lambda settings: CustomEvaluator()
    }

    @classmethod
    def register(cls, backend: str, builder: EvaluatorBuilder) -> None:
        """Register an evaluator builder by backend name."""
        normalized = cls._normalize_backend(backend)
        if not callable(builder):
            raise TypeError("Evaluator builder must be callable")
        cls._backends[normalized] = builder

    @classmethod
    def unregister(cls, backend: str) -> None:
        """Remove a backend registration if it exists."""
        normalized = cls._normalize_backend(backend)
        if normalized == "custom":
            cls._backends[normalized] = lambda settings: CustomEvaluator()
            return
        cls._backends.pop(normalized, None)

    @classmethod
    def reset_defaults(cls) -> None:
        """Reset registrations to built-in defaults."""
        cls._backends = {"custom": lambda settings: CustomEvaluator()}

    @classmethod
    def create(
        cls,
        settings: Settings | EvaluationSettings,
        backend: str | None = None,
    ) -> BaseEvaluator:
        """Create an evaluator from settings.

        When backend is omitted, the first configured backend is used.
        """
        evaluation_settings = settings.evaluation if isinstance(settings, Settings) else settings
        selected_backend = backend or _first_backend(evaluation_settings)
        normalized = cls._normalize_backend(selected_backend)

        try:
            builder = cls._backends[normalized]
        except KeyError as exc:
            available = ", ".join(sorted(cls._backends)) or "none"
            raise EvaluatorFactoryError(
                f"Unknown evaluator backend: {selected_backend}. "
                f"Registered backends: {available}"
            ) from exc

        evaluator = builder(evaluation_settings)
        if not isinstance(evaluator, BaseEvaluator):
            raise EvaluatorFactoryError(
                f"Evaluator backend '{normalized}' returned {type(evaluator).__name__}, "
                "expected BaseEvaluator"
            )
        return evaluator

    @staticmethod
    def _normalize_backend(backend: str) -> str:
        if not backend or not backend.strip():
            raise EvaluatorFactoryError("Evaluator backend name is required")
        return backend.strip().lower()


def _first_backend(settings: EvaluationSettings) -> str:
    if not settings.backends:
        raise EvaluatorFactoryError("At least one evaluator backend is required")
    return settings.backends[0]
