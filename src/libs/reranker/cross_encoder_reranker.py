"""Cross-encoder-style reranker implementation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Protocol

from core.settings import RerankSettings
from libs.reranker.base_reranker import BaseReranker, RerankCandidate, RerankResult


class CrossEncoderRerankerError(ValueError):
    """Raised when cross-encoder scoring cannot produce usable scores."""


class CrossEncoderScorer(Protocol):
    """Callable scoring adapter for local or hosted cross-encoder models."""

    def score(self, query: str, candidates: list[RerankCandidate]) -> Sequence[float]:
        """Return one relevance score per candidate."""


class KeywordOverlapScorer:
    """Small deterministic scorer used as the default runnable placeholder."""

    def score(self, query: str, candidates: list[RerankCandidate]) -> list[float]:
        query_terms = _tokenize(query)
        if not query_terms:
            return [0.0 for _ in candidates]

        query_set = set(query_terms)
        scores: list[float] = []
        for candidate in candidates:
            candidate_terms = _tokenize(candidate.text)
            candidate_set = set(candidate_terms)
            overlap = len(query_set & candidate_set)
            density = overlap / max(len(candidate_set), 1)
            scores.append(float(overlap) + density)
        return scores


class CrossEncoderReranker(BaseReranker):
    """Rerank top-M candidates with a pairwise query/document scorer."""

    def __init__(
        self,
        settings: RerankSettings,
        scorer: CrossEncoderScorer | None = None,
    ) -> None:
        self.settings = settings
        self.scorer = scorer or KeywordOverlapScorer()

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Any | None = None,
    ) -> list[RerankResult]:
        if not isinstance(query, str) or not query.strip():
            raise CrossEncoderRerankerError("query must be a non-empty string")
        if not candidates:
            return []

        rerank_count = _bounded_top_m(self.settings.top_m, len(candidates))
        head = candidates[:rerank_count]
        tail = candidates[rerank_count:]
        scores = _validate_scores(self.scorer.score(query, head), len(head))

        indexed = list(zip(head, scores, range(len(head))))
        indexed.sort(key=lambda item: (-item[1], item[2]))

        results: list[RerankResult] = []
        for candidate, score, _ in indexed:
            results.append(_to_result(candidate, score, len(results) + 1))
        for candidate in tail:
            results.append(_to_result(candidate, candidate.score, len(results) + 1))
        return results


def _bounded_top_m(top_m: int, total: int) -> int:
    if top_m <= 0:
        return total
    return min(top_m, total)


def _validate_scores(scores: Sequence[float], expected_count: int) -> list[float]:
    if len(scores) != expected_count:
        raise CrossEncoderRerankerError(
            f"cross-encoder scorer returned {len(scores)} scores for {expected_count} candidates"
        )

    validated: list[float] = []
    for score in scores:
        try:
            validated.append(float(score))
        except (TypeError, ValueError) as exc:
            raise CrossEncoderRerankerError("cross-encoder scorer must return numeric scores") from exc
    return validated


def _to_result(candidate: RerankCandidate, score: float, rank: int) -> RerankResult:
    return RerankResult(
        id=candidate.id,
        text=candidate.text,
        score=score,
        rank=rank,
        metadata=candidate.metadata,
    )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower())
