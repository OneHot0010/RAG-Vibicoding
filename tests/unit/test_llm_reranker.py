"""Tests for the LLM-backed reranker."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Mapping

import pytest

from core.settings import RerankSettings
from libs.llm.base_llm import BaseLLM, ChatMessage
from libs.reranker import LLMReranker, LLMRerankerError, RerankCandidate, RerankerFactory


class FakeLLM(BaseLLM):
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: Sequence[ChatMessage | Mapping[str, Any]] | None = None

    def chat(self, messages: Sequence[ChatMessage | Mapping[str, Any]]) -> str:
        self.messages = messages
        return self.response


def candidates() -> list[RerankCandidate]:
    return [
        RerankCandidate(id="a", text="alpha", score=0.1, metadata={"source": "a.md"}),
        RerankCandidate(id="b", text="beta", score=0.2, metadata={"source": "b.md"}),
        RerankCandidate(id="c", text="gamma", score=0.3, metadata={"source": "c.md"}),
    ]


def test_llm_reranker_orders_by_ranked_ids() -> None:
    llm = FakeLLM(json.dumps({"ranked_ids": ["b", "a", "c"]}))
    reranker = LLMReranker(RerankSettings(backend="llm"), llm, prompt_template="Rank JSON.")

    results = reranker.rerank("query", candidates())

    assert [result.id for result in results] == ["b", "a", "c"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.score for result in results] == [3.0, 2.0, 1.0]
    assert results[0].metadata == {"source": "b.md"}


def test_llm_reranker_preserves_unreturned_candidates_at_tail() -> None:
    llm = FakeLLM(json.dumps({"ranked_ids": ["c"]}))
    reranker = LLMReranker(RerankSettings(backend="llm"), llm, prompt_template="Rank JSON.")

    results = reranker.rerank("query", candidates())

    assert [result.id for result in results] == ["c", "a", "b"]


def test_prompt_contains_query_candidates_and_schema() -> None:
    llm = FakeLLM(json.dumps({"ranked_ids": ["a"]}))
    reranker = LLMReranker(RerankSettings(backend="llm"), llm, prompt_template="Rank JSON.")

    reranker.rerank("needle", candidates())

    assert llm.messages is not None
    assert isinstance(llm.messages[0], ChatMessage)
    assert llm.messages[0].content == "Rank JSON."
    assert isinstance(llm.messages[1], ChatMessage)
    payload = json.loads(llm.messages[1].content)
    assert payload["query"] == "needle"
    assert payload["candidates"][0]["id"] == "a"
    assert payload["required_output_schema"] == {"ranked_ids": ["candidate-id"]}


def test_empty_candidates_return_empty_without_llm_call() -> None:
    llm = FakeLLM(json.dumps({"ranked_ids": []}))
    reranker = LLMReranker(RerankSettings(backend="llm"), llm, prompt_template="Rank JSON.")

    assert reranker.rerank("query", []) == []
    assert llm.messages is None


def test_invalid_json_response_has_readable_error() -> None:
    reranker = LLMReranker(RerankSettings(backend="llm"), FakeLLM("not json"), prompt_template="Rank JSON.")

    with pytest.raises(LLMRerankerError, match="valid JSON"):
        reranker.rerank("query", candidates())


def test_invalid_schema_response_has_readable_error() -> None:
    reranker = LLMReranker(
        RerankSettings(backend="llm"),
        FakeLLM(json.dumps({"ids": ["a"]})),
        prompt_template="Rank JSON.",
    )

    with pytest.raises(LLMRerankerError, match="ranked_ids"):
        reranker.rerank("query", candidates())


def test_duplicate_ranked_ids_have_readable_error() -> None:
    reranker = LLMReranker(
        RerankSettings(backend="llm"),
        FakeLLM(json.dumps({"ranked_ids": ["a", "a"]})),
        prompt_template="Rank JSON.",
    )

    with pytest.raises(LLMRerankerError, match="duplicates"):
        reranker.rerank("query", candidates())


def test_unknown_candidate_id_has_readable_error() -> None:
    reranker = LLMReranker(
        RerankSettings(backend="llm"),
        FakeLLM(json.dumps({"ranked_ids": ["missing"]})),
        prompt_template="Rank JSON.",
    )

    with pytest.raises(LLMRerankerError, match="unknown candidate ids"):
        reranker.rerank("query", candidates())


def test_prompt_can_be_loaded_from_file(tmp_path) -> None:
    prompt_path = tmp_path / "rerank.txt"
    prompt_path.write_text("Use ranked_ids.", encoding="utf-8")

    reranker = LLMReranker(
        RerankSettings(backend="llm"),
        FakeLLM(json.dumps({"ranked_ids": ["a"]})),
        prompt_path=prompt_path,
    )

    assert reranker.prompt_template == "Use ranked_ids."


def test_factory_can_create_registered_llm_reranker() -> None:
    RerankerFactory.reset_defaults()
    fake_llm = FakeLLM(json.dumps({"ranked_ids": ["a"]}))
    RerankerFactory.register(
        "llm",
        lambda settings: LLMReranker(settings, fake_llm, prompt_template="Rank JSON."),
    )

    reranker = RerankerFactory.create(RerankSettings(backend="llm"))

    assert isinstance(reranker, LLMReranker)
