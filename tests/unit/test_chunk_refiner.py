"""Tests for chunk refinement transform."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

import pytest

from core.settings import (
    EmbeddingSettings,
    EvaluationSettings,
    LLMSettings,
    ObservabilitySettings,
    RerankSettings,
    RetrievalSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
    VisionLLMSettings,
)
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform import BaseTransform, ChunkRefiner, TransformError
from libs.llm import BaseLLM, ChatMessage


class FakeLLM(BaseLLM):
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.messages: Sequence[ChatMessage | Mapping[str, Any]] | None = None

    def chat(self, messages: Sequence[ChatMessage | Mapping[str, Any]]) -> str:
        self.messages = messages
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def make_settings(use_llm: bool = False) -> Settings:
    return Settings(
        llm=LLMSettings(provider="fake", model="fake-chat"),
        embedding=EmbeddingSettings(provider="fake", model="fake-embedding"),
        vision_llm=VisionLLMSettings(provider="fake", model="fake-vision"),
        splitter=SplitterSettings(strategy="fake", chunk_size=100, chunk_overlap=0),
        vector_store=VectorStoreSettings(backend="fake", persist_path="./tmp/vector"),
        retrieval=RetrievalSettings("bm25", "rrf", 20, 20, 10),
        rerank=RerankSettings(backend="none"),
        evaluation=EvaluationSettings(backends=["custom"], golden_test_set="./golden.json"),
        observability=ObservabilitySettings(enabled=True, log_file="./logs/traces.jsonl"),
        raw={"ingestion": {"chunk_refiner": {"use_llm": use_llm}}},
    )


def make_chunk(text: str = "Header\n\nUseful text.\n\nPage 1") -> Chunk:
    return Chunk(
        id="chunk-1",
        text=text,
        metadata={"source_path": "docs/a.md", "chunk_index": 0},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )


def test_base_transform_contract_can_be_implemented() -> None:
    class IdentityTransform(BaseTransform):
        def transform(self, chunks: list[Chunk], trace: Any | None = None) -> list[Chunk]:
            return chunks

    chunk = make_chunk("Clean")

    assert IdentityTransform().transform([chunk]) == [chunk]


@pytest.mark.parametrize("case", json.loads(Path("tests/fixtures/noisy_chunks.json").read_text(encoding="utf-8")))
def test_rule_based_refine_cleans_fixture_noise(case: dict[str, str]) -> None:
    refiner = ChunkRefiner(make_settings())

    result = refiner._rule_based_refine(case["input"])

    assert case["expected_contains"] in result
    assert "<!--" not in result
    assert "Page 1 of 3" not in result


def test_rule_based_refine_preserves_code_block_spacing() -> None:
    text = "Intro\n\n```python\nx  =  1\nprint(x)\n```\n\nFooter"
    refiner = ChunkRefiner(make_settings())

    result = refiner._rule_based_refine(text)

    assert "```python\nx  =  1\nprint(x)\n```" in result


def test_transform_rule_mode_updates_text_and_metadata() -> None:
    chunk = make_chunk()

    result = ChunkRefiner(make_settings()).transform([chunk])[0]

    assert result.text == "Useful text."
    assert result.metadata["refined_by"] == "rule"
    assert result.id == chunk.id
    assert result.source_ref == chunk.source_ref


def test_llm_mode_calls_llm_and_marks_metadata() -> None:
    llm = FakeLLM("Refined by LLM")
    chunk = make_chunk(" noisy text ")

    result = ChunkRefiner(make_settings(use_llm=True), llm=llm).transform([chunk])[0]

    assert result.text == "Refined by LLM"
    assert result.metadata["refined_by"] == "llm"
    assert llm.messages is not None
    assert isinstance(llm.messages[1], ChatMessage)
    assert "noisy text" in llm.messages[1].content


def test_llm_failure_falls_back_to_rule_result() -> None:
    llm = FakeLLM(RuntimeError("network down"))
    chunk = make_chunk()

    result = ChunkRefiner(make_settings(use_llm=True), llm=llm).transform([chunk])[0]

    assert result.text == "Useful text."
    assert result.metadata["refined_by"] == "rule"
    assert result.metadata["fallback_reason"] == "llm_refine_failed"


def test_empty_llm_response_falls_back_to_rule_result() -> None:
    llm = FakeLLM("   ")

    result = ChunkRefiner(make_settings(use_llm=True), llm=llm).transform([make_chunk()])[0]

    assert result.metadata["fallback_reason"] == "llm_refine_failed"
    assert result.text == "Useful text."


def test_prompt_loader_adds_text_placeholder_when_missing(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Rewrite safely.", encoding="utf-8")

    refiner = ChunkRefiner(make_settings(), prompt_path=prompt_path)

    assert "{text}" in refiner.prompt_template


def test_prompt_loader_uses_fallback_when_missing(tmp_path: Path) -> None:
    refiner = ChunkRefiner(make_settings(), prompt_path=tmp_path / "missing.txt")

    assert "{text}" in refiner.prompt_template


def test_trace_context_records_refinement_stages() -> None:
    trace = TraceContext()

    ChunkRefiner(make_settings()).transform([make_chunk()], trace=trace)

    assert trace.trace_id
    assert trace.stages == [{"name": "chunk_refiner.rule", "data": {"chunk_id": "chunk-1"}}]


def test_single_chunk_exception_does_not_block_other_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    refiner = ChunkRefiner(make_settings())

    def fake_rule(text: str) -> str:
        if "bad" in text:
            raise TransformError("bad chunk")
        return text.strip()

    monkeypatch.setattr(refiner, "_rule_based_refine", fake_rule)
    bad = make_chunk("bad")
    good = Chunk(
        id="chunk-2",
        text=" good ",
        metadata={"source_path": "docs/a.md", "chunk_index": 1},
        start_offset=5,
        end_offset=11,
        source_ref="doc-1",
    )

    results = refiner.transform([bad, good])

    assert results[0].text == "bad"
    assert results[0].metadata["refined_by"] == "original"
    assert results[0].metadata["refine_error"] == "bad chunk"
    assert results[1].text == "good"


def test_non_string_text_has_readable_error() -> None:
    refiner = ChunkRefiner(make_settings())

    with pytest.raises(TransformError, match="string"):
        refiner._rule_based_refine(None)  # type: ignore[arg-type]
