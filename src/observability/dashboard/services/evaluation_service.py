"""Dashboard evaluation service wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.settings import Settings, load_settings
from libs.evaluator import EvaluatorFactory
from observability.evaluation import EvalReport, EvalRunner, load_golden_test_set


class EvaluationService:
    """Build and run evaluation from dashboard controls."""

    def __init__(
        self,
        *,
        settings_path: str | Path = "config/settings.yaml",
        data_dir: str | Path = "data",
        query_components_builder: Any | None = None,
        evaluator_factory: Any | None = None,
    ) -> None:
        self.settings_path = Path(settings_path)
        self.data_dir = Path(data_dir)
        self.query_components_builder = query_components_builder or _default_query_components_builder
        self.evaluator_factory = evaluator_factory or EvaluatorFactory

    def load_settings(self) -> Settings:
        """Load project settings."""
        return load_settings(self.settings_path)

    def available_backends(self) -> list[str]:
        """Return configured evaluator choices plus an all option."""
        backends = list(self.load_settings().evaluation.backends)
        return ["all", *backends] if len(backends) > 1 else backends

    def default_test_set_path(self) -> str:
        """Return configured golden test set path."""
        return self.load_settings().evaluation.golden_test_set

    def test_set_summary(self, test_set_path: str | Path | None = None) -> dict[str, Any]:
        """Return a small summary of a golden test set."""
        path = Path(test_set_path or self.default_test_set_path())
        cases = load_golden_test_set(path)
        return {
            "path": str(path),
            "case_count": len(cases),
            "queries": [case.query for case in cases],
        }

    def run_evaluation(
        self,
        *,
        test_set_path: str | Path | None = None,
        backend: str | None = None,
        top_k: int | None = None,
        online_embedding: bool = False,
    ) -> dict[str, Any]:
        """Run evaluation and return a serialized report."""
        settings = self.load_settings()
        selected_backend = None if backend in {None, "", "all"} else backend
        path = test_set_path or settings.evaluation.golden_test_set
        components = self.query_components_builder(settings, self.data_dir, offline_embedding=not online_embedding)
        evaluator = self.evaluator_factory.create(settings, backend=selected_backend)
        report: EvalReport = EvalRunner(
            settings,
            hybrid_search=components.hybrid_search,
            evaluator=evaluator,
            top_k=top_k,
        ).run(path)
        return report.to_dict()


def _default_query_components_builder(settings: Settings, data_dir: str | Path, offline_embedding: bool = True) -> Any:
    from scripts.query import build_query_components

    return build_query_components(settings, data_dir, offline_embedding=offline_embedding)


__all__ = ["EvaluationService"]
