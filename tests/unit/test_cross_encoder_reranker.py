"""Tests for the cross-encoder-style reranker."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from core.settings import RerankSettings
from libs.reranker import (
    CrossEncoderReranker,
    CrossEncoderRerankerError,
    RerankCandidate,
    RerankerFactory,
)


class FakeScorer:
    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = list(scores)
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, candidates: list[RerankCandidate]) -> Sequence[float]:
        self.calls.append((query, [candidate.id for candidate in candidates]))
        return self.scores


def candidates() -> list[RerankCandidate]:
    return [
        RerankCandidate(id="a", text="alpha chunk", score=0.1, metadata={"source": "a.md"}),
        RerankCandidate(id="b", text="beta chunk", score=0.2, metadata={"source": "b.md"}),
        RerankCandidate(id="c", text="gamma chunk", score=0.3, metadata={"source": "c.md"}),
        RerankCandidate(id="d", text="delta chunk", score=0.4, metadata={"source": "d.md"}),
    ]


def test_cross_encoder_orders_by_mock_scores() -> None:
    reranker = CrossEncoderReranker(
        RerankSettings(backend="cross_encoder"),
        scorer=FakeScorer([0.2, 0.9, 0.1, 0.5]),
    )

    results = reranker.rerank("query", candidates())

    assert [result.id for result in results] == ["b", "d", "a", "c"]
    assert [result.score for result in results] == [0.9, 0.5, 0.2, 0.1]
    assert [result.rank for result in results] == [1, 2, 3, 4]
    assert results[0].metadata == {"source": "b.md"}


def test_cross_encoder_only_scores_top_m_and_preserves_tail() -> None:
    scorer = FakeScorer([0.1, 0.9])
    reranker = CrossEncoderReranker(
        RerankSettings(backend="cross_encoder", top_m=2),
        scorer=scorer,
    )

    results = reranker.rerank("needle", candidates())

    assert scorer.calls == [("needle", ["a", "b"])]
    assert [result.id for result in results] == ["b", "a", "c", "d"]
    assert [result.score for result in results] == [0.9, 0.1, 0.3, 0.4]


def test_tied_scores_keep_original_order() -> None:
    reranker = CrossEncoderReranker(
        RerankSettings(backend="cross_encoder"),
        scorer=FakeScorer([1.0, 1.0, 0.0, 1.0]),
    )

    results = reranker.rerank("query", candidates())

    assert [result.id for result in results] == ["a", "b", "d", "c"]


def test_empty_candidates_return_empty_without_scoring() -> None:
    scorer = FakeScorer([])
    reranker = CrossEncoderReranker(RerankSettings(backend="cross_encoder"), scorer=scorer)

    assert reranker.rerank("query", []) == []
    assert scorer.calls == []


def test_default_keyword_overlap_scorer_is_runnable() -> None:
    reranker = CrossEncoderReranker(RerankSettings(backend="cross_encoder"))
    items = [
        RerankCandidate(id="low", text="unrelated text"),
        RerankCandidate(id="high", text="rag rerank query"),
    ]

    results = reranker.rerank("rag query", items)

    assert [result.id for result in results] == ["high", "low"]


def test_query_must_be_non_empty() -> None:
    reranker = CrossEncoderReranker(RerankSettings(backend="cross_encoder"), scorer=FakeScorer([]))

    with pytest.raises(CrossEncoderRerankerError, match="non-empty"):
        reranker.rerank(" ", candidates())


def test_score_count_mismatch_has_readable_error() -> None:
    reranker = CrossEncoderReranker(
        RerankSettings(backend="cross_encoder"),
        scorer=FakeScorer([1.0]),
    )

    with pytest.raises(CrossEncoderRerankerError, match="1 scores for 4 candidates"):
        reranker.rerank("query", candidates())


def test_non_numeric_score_has_readable_error() -> None:
    reranker = CrossEncoderReranker(
        RerankSettings(backend="cross_encoder"),
        scorer=FakeScorer([1.0, "bad", 0.5, 0.0]),  # type: ignore[list-item]
    )

    with pytest.raises(CrossEncoderRerankerError, match="numeric scores"):
        reranker.rerank("query", candidates())


def test_factory_can_create_cross_encoder_reranker() -> None:
    RerankerFactory.reset_defaults()

    reranker = RerankerFactory.create(RerankSettings(backend="cross_encoder", model="mock-model"))

    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.settings.model == "mock-model"
