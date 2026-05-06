"""Tests for the settings loader and fail-fast validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.settings import Settings, SettingsError, load_settings


VALID_SETTINGS = """
llm:
  provider: azure
  model: gpt-4o
embedding:
  provider: openai
  model: text-embedding-3-small
vision_llm:
  provider: azure
  model: gpt-4o
vector_store:
  backend: chroma
  persist_path: ./data/db/chroma
retrieval:
  sparse_backend: bm25
  fusion_algorithm: rrf
  top_k_dense: 20
  top_k_sparse: 20
  top_k_final: 10
rerank:
  backend: none
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_m: 30
evaluation:
  backends:
    - custom
  golden_test_set: ./tests/fixtures/golden_test_set.json
observability:
  enabled: true
  log_file: ./logs/traces.jsonl
dashboard:
  enabled: true
  port: 8501
  traces_dir: ./logs
  auto_refresh: true
  refresh_interval: 5
"""


def test_load_default_settings_file() -> None:
    settings = load_settings()

    assert isinstance(settings, Settings)
    assert settings.llm.provider == "azure"
    assert settings.embedding.provider == "openai"
    assert settings.vector_store.backend == "chroma"
    assert settings.retrieval.top_k_final == 10
    assert settings.evaluation.backends == ["custom"]


def test_load_settings_from_explicit_path(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(VALID_SETTINGS, encoding="utf-8")

    settings = load_settings(settings_path)

    assert settings.rerank.backend == "none"
    assert settings.dashboard is not None
    assert settings.dashboard.port == 8501


def test_missing_required_field_names_field_path(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        VALID_SETTINGS.replace("  provider: openai\n", "", 1),
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="embedding.provider"):
        load_settings(settings_path)


def test_invalid_numeric_field_names_field_path(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        VALID_SETTINGS.replace("  top_k_final: 10", "  top_k_final: many"),
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="retrieval.top_k_final"):
        load_settings(settings_path)


def test_missing_settings_file_raises_readable_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(SettingsError, match="Settings file not found"):
        load_settings(missing_path)
