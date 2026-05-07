"""Tests for metadata enrichment transform."""

from __future__ import annotations

import json
from collections.abc import Sequence
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
from ingestion.transform import MetadataEnricher
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
        raw={"ingestion": {"metadata_enricher": {"use_llm": use_llm}}},
    )


def make_chunk(text: str = "Hybrid retrieval combines dense vectors with BM25 sparse search.") -> Chunk:
    return Chunk(
        id="chunk-1",
        text=text,
        metadata={"source_path": "docs/a.md", "chunk_index": 0},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )


def test_rule_mode_adds_non_empty_title_summary_and_tags() -> None:
    result = MetadataEnricher(make_settings()).transform([make_chunk()])[0]

    assert result.metadata["title"]
    assert result.metadata["summary"]
    assert result.metadata["tags"]
    assert result.metadata["metadata_enriched_by"] == "rule"
    assert "hybrid" in result.metadata["tags"]


def test_rule_mode_preserves_existing_metadata_and_text() -> None:
    chunk = make_chunk()

    result = MetadataEnricher(make_settings()).transform([chunk])[0]

    assert result.text == chunk.text
    assert result.metadata["source_path"] == "docs/a.md"
    assert result.metadata["chunk_index"] == 0
    assert result.source_ref == "doc-1"


def test_rule_mode_handles_empty_text_with_fallback_values() -> None:
    chunk = make_chunk("")

    result = MetadataEnricher(make_settings()).transform([chunk])[0]

    assert result.metadata["title"] == "Untitled Chunk"
    assert result.metadata["summary"] == "No summary available."
    assert result.metadata["tags"] == ["general"]


def test_llm_mode_uses_structured_metadata_and_marks_llm() -> None:
    llm = FakeLLM(
        json.dumps(
            {
                "title": "Hybrid Retrieval",
                "summary": "Combines dense and sparse search.",
                "tags": ["retrieval", "bm25", "dense"],
            }
        )
    )

    result = MetadataEnricher(make_settings(use_llm=True), llm=llm).transform([make_chunk()])[0]

    assert result.metadata["title"] == "Hybrid Retrieval"
    assert result.metadata["summary"] == "Combines dense and sparse search."
    assert result.metadata["tags"] == ["retrieval", "bm25", "dense"]
    assert result.metadata["metadata_enriched_by"] == "llm"
    assert llm.messages is not None
    assert isinstance(llm.messages[1], ChatMessage)
    assert "Hybrid retrieval" in llm.messages[1].content


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        json.dumps({"title": "", "summary": "summary", "tags": ["x"]}),
        json.dumps({"title": "Title", "summary": "", "tags": ["x"]}),
        json.dumps({"title": "Title", "summary": "summary", "tags": []}),
    ],
)
def test_invalid_llm_output_falls_back_to_rule_metadata(response: str) -> None:
    result = MetadataEnricher(make_settings(use_llm=True), llm=FakeLLM(response)).transform([make_chunk()])[0]

    assert result.metadata["metadata_enriched_by"] == "rule"
    assert result.metadata["metadata_fallback_reason"] == "llm_enrichment_failed"
    assert result.metadata["title"]
    assert result.metadata["summary"]
    assert result.metadata["tags"]


def test_llm_exception_falls_back_to_rule_metadata() -> None:
    result = MetadataEnricher(
        make_settings(use_llm=True),
        llm=FakeLLM(RuntimeError("provider failed")),
    ).transform([make_chunk()])[0]

    assert result.metadata["metadata_enriched_by"] == "rule"
    assert result.metadata["metadata_fallback_reason"] == "llm_enrichment_failed"


def test_trace_records_rule_and_llm_paths() -> None:
    rule_trace = TraceContext()
    MetadataEnricher(make_settings()).transform([make_chunk()], trace=rule_trace)

    llm_trace = TraceContext()
    MetadataEnricher(
        make_settings(use_llm=True),
        llm=FakeLLM(json.dumps({"title": "T", "summary": "S", "tags": ["tag"]})),
    ).transform([make_chunk()], trace=llm_trace)

    assert rule_trace.stages == [{"name": "metadata_enricher.rule", "data": {"chunk_id": "chunk-1"}}]
    assert llm_trace.stages == [{"name": "metadata_enricher.llm", "data": {"chunk_id": "chunk-1"}}]


def test_single_chunk_exception_does_not_block_other_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    enricher = MetadataEnricher(make_settings())

    def fake_llm_enrich(text: str, trace: Any | None = None) -> dict[str, Any] | None:
        if "bad" in text:
            raise RuntimeError("bad chunk")
        return None

    monkeypatch.setattr(enricher, "_use_llm", lambda: True)
    monkeypatch.setattr(enricher, "_llm_enrich", fake_llm_enrich)

    bad = make_chunk("bad")
    good = Chunk(
        id="chunk-2",
        text="good retrieval text",
        metadata={"source_path": "docs/a.md", "chunk_index": 1},
        start_offset=0,
        end_offset=19,
    )
    results = enricher.transform([bad, good])

    assert results[0].metadata["metadata_enrichment_error"] == "bad chunk"
    assert results[1].metadata["metadata_enriched_by"] == "rule"


def test_metadata_enriched_chunks_are_serializable() -> None:
    result = MetadataEnricher(make_settings()).transform([make_chunk()])[0]

    assert Chunk.from_dict(result.to_dict()) == result
