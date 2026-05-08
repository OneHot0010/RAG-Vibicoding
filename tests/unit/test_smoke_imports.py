"""Smoke tests for the scaffolded package layout."""

from __future__ import annotations

import importlib
from pathlib import Path


TOP_LEVEL_PACKAGES = (
    "mcp_server",
    "core",
    "ingestion",
    "libs",
    "observability",
)

REPRESENTATIVE_MODULES = (
    "mcp_server.server",
    "core.settings",
    "core.types",
    "ingestion.document_manager",
    "ingestion.pipeline",
    "libs.llm.base_llm",
    "libs.embedding.base_embedding",
    "observability.logger",
    "observability.dashboard.app",
    "observability.dashboard.pages.overview",
    "observability.dashboard.services.config_service",
    "observability.dashboard.services.data_service",
)


def test_top_level_packages_import() -> None:
    for package_name in TOP_LEVEL_PACKAGES:
        assert importlib.import_module(package_name)


def test_representative_modules_import() -> None:
    for module_name in REPRESENTATIVE_MODULES:
        assert importlib.import_module(module_name)


def test_sample_document_fixture_exists() -> None:
    sample = Path("tests/fixtures/sample_documents/minimal.md")

    assert sample.is_file()
    assert sample.read_text(encoding="utf-8").strip()
