"""Configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_SETTINGS_PATH = Path("config/settings.yaml")


class SettingsError(ValueError):
    """Raised when project settings are missing or invalid."""


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    azure_endpoint: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    api_version: str | None = None
    deployment: str | None = None


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str
    model: str


@dataclass(frozen=True)
class VisionLLMSettings:
    provider: str
    model: str


@dataclass(frozen=True)
class SplitterSettings:
    strategy: str
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass(frozen=True)
class VectorStoreSettings:
    backend: str
    persist_path: str


@dataclass(frozen=True)
class RetrievalSettings:
    sparse_backend: str
    fusion_algorithm: str
    top_k_dense: int
    top_k_sparse: int
    top_k_final: int


@dataclass(frozen=True)
class RerankSettings:
    backend: str
    model: str | None = None
    top_m: int = 0


@dataclass(frozen=True)
class EvaluationSettings:
    backends: list[str]
    golden_test_set: str


@dataclass(frozen=True)
class ObservabilitySettings:
    enabled: bool
    log_file: str


@dataclass(frozen=True)
class DashboardSettings:
    enabled: bool
    port: int
    traces_dir: str
    auto_refresh: bool
    refresh_interval: int


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings
    embedding: EmbeddingSettings
    vision_llm: VisionLLMSettings
    splitter: SplitterSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings
    dashboard: DashboardSettings | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def load_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> Settings:
    """Read a YAML settings file, parse it into dataclasses, and validate it."""
    settings_path = Path(path)
    if not settings_path.exists():
        raise SettingsError(f"Settings file not found: {settings_path}")

    data = _load_yaml(settings_path)
    if not isinstance(data, dict):
        raise SettingsError("Settings file must contain a mapping at the top level")

    data = _expand_env_vars(data)
    settings = _parse_settings(data)
    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    """Validate required settings and surface readable field paths."""
    required_fields = {
        "llm.provider": settings.llm.provider,
        "llm.model": settings.llm.model,
        "embedding.provider": settings.embedding.provider,
        "embedding.model": settings.embedding.model,
        "splitter.strategy": settings.splitter.strategy,
        "splitter.chunk_size": settings.splitter.chunk_size,
        "splitter.chunk_overlap": settings.splitter.chunk_overlap,
        "vector_store.backend": settings.vector_store.backend,
        "vector_store.persist_path": settings.vector_store.persist_path,
        "retrieval.sparse_backend": settings.retrieval.sparse_backend,
        "retrieval.fusion_algorithm": settings.retrieval.fusion_algorithm,
        "retrieval.top_k_dense": settings.retrieval.top_k_dense,
        "retrieval.top_k_sparse": settings.retrieval.top_k_sparse,
        "retrieval.top_k_final": settings.retrieval.top_k_final,
        "rerank.backend": settings.rerank.backend,
        "evaluation.backends": settings.evaluation.backends,
        "evaluation.golden_test_set": settings.evaluation.golden_test_set,
        "observability.enabled": settings.observability.enabled,
        "observability.log_file": settings.observability.log_file,
    }

    for field_path, value in required_fields.items():
        if value is None or value == "" or value == []:
            raise SettingsError(f"Missing required setting: {field_path}")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SettingsError("PyYAML is required to read config/settings.yaml") from exc

    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    return loaded or {}


def _parse_settings(data: dict[str, Any]) -> Settings:
    return Settings(
        llm=LLMSettings(
            provider=_required(data, "llm.provider"),
            model=_required(data, "llm.model"),
            azure_endpoint=_optional(data, "llm.azure_endpoint"),
            api_key=_optional(data, "llm.api_key"),
            base_url=_optional(data, "llm.base_url"),
            api_version=_optional(data, "llm.api_version"),
            deployment=_optional(data, "llm.deployment"),
        ),
        embedding=EmbeddingSettings(
            provider=_required(data, "embedding.provider"),
            model=_required(data, "embedding.model"),
        ),
        vision_llm=VisionLLMSettings(
            provider=_required(data, "vision_llm.provider"),
            model=_required(data, "vision_llm.model"),
        ),
        splitter=SplitterSettings(
            strategy=_required(data, "splitter.strategy"),
            chunk_size=_required_int(data, "splitter.chunk_size"),
            chunk_overlap=_required_int(data, "splitter.chunk_overlap"),
        ),
        vector_store=VectorStoreSettings(
            backend=_required(data, "vector_store.backend"),
            persist_path=_required(data, "vector_store.persist_path"),
        ),
        retrieval=RetrievalSettings(
            sparse_backend=_required(data, "retrieval.sparse_backend"),
            fusion_algorithm=_required(data, "retrieval.fusion_algorithm"),
            top_k_dense=_required_int(data, "retrieval.top_k_dense"),
            top_k_sparse=_required_int(data, "retrieval.top_k_sparse"),
            top_k_final=_required_int(data, "retrieval.top_k_final"),
        ),
        rerank=RerankSettings(
            backend=_required(data, "rerank.backend"),
            model=_optional(data, "rerank.model"),
            top_m=_optional_int(data, "rerank.top_m", default=0),
        ),
        evaluation=EvaluationSettings(
            backends=_required_list(data, "evaluation.backends"),
            golden_test_set=_required(data, "evaluation.golden_test_set"),
        ),
        observability=ObservabilitySettings(
            enabled=_required_bool(data, "observability.enabled"),
            log_file=_required(data, "observability.log_file"),
        ),
        dashboard=_parse_dashboard(data),
        raw=data,
    )


def _parse_dashboard(data: dict[str, Any]) -> DashboardSettings | None:
    dashboard = _optional(data, "dashboard")
    if dashboard is None:
        return None
    if not isinstance(dashboard, dict):
        raise SettingsError("Setting dashboard must be a mapping")
    return DashboardSettings(
        enabled=_required_bool(data, "dashboard.enabled"),
        port=_required_int(data, "dashboard.port"),
        traces_dir=_required(data, "dashboard.traces_dir"),
        auto_refresh=_required_bool(data, "dashboard.auto_refresh"),
        refresh_interval=_required_int(data, "dashboard.refresh_interval"),
    )


def _required(data: dict[str, Any], path: str) -> Any:
    value = _optional(data, path)
    if value is None or value == "":
        raise SettingsError(f"Missing required setting: {path}")
    return value


def _optional(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _required_int(data: dict[str, Any], path: str) -> int:
    value = _required(data, path)
    if isinstance(value, bool):
        raise SettingsError(f"Setting {path} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SettingsError(f"Setting {path} must be an integer") from exc


def _optional_int(data: dict[str, Any], path: str, default: int) -> int:
    value = _optional(data, path)
    if value is None:
        return default
    if isinstance(value, bool):
        raise SettingsError(f"Setting {path} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SettingsError(f"Setting {path} must be an integer") from exc


def _required_bool(data: dict[str, Any], path: str) -> bool:
    value = _required(data, path)
    if not isinstance(value, bool):
        raise SettingsError(f"Setting {path} must be a boolean")
    return value


def _required_list(data: dict[str, Any], path: str) -> list[str]:
    value = _required(data, path)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SettingsError(f"Setting {path} must be a list of strings")
    return value


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    return value
