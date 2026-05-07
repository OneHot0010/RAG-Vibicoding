"""Optional real-LLM checks for chunk refinement.

Run with RUN_REAL_LLM_TESTS=1 and valid provider credentials when doing manual acceptance.
"""

from __future__ import annotations

import os

import pytest

from core.settings import load_settings
from core.types import Chunk
from ingestion.transform import ChunkRefiner


pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.getenv("RUN_REAL_LLM_TESTS") != "1", reason="real LLM acceptance is opt-in")
def test_chunk_refiner_real_llm_acceptance() -> None:
    settings = load_settings()
    settings.raw.setdefault("ingestion", {}).setdefault("chunk_refiner", {})["use_llm"] = True
    chunk = Chunk(
        id="chunk-real",
        text="Header\n\nThis chunk explains retrieval quality.      Page 1\n\nFooter",
        metadata={"source_path": "docs/a.md", "chunk_index": 0},
        start_offset=0,
        end_offset=64,
    )

    result = ChunkRefiner(settings).transform([chunk])[0]

    assert result.text
    assert "retrieval quality" in result.text.lower()
    assert result.metadata["refined_by"] in {"llm", "rule"}
