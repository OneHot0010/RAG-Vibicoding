"""Core-level reranker orchestration with safe fallback."""

from __future__ import annotations

import copy
from typing import Any

from core.settings import Settings
from core.types import RetrievalResult
from libs.reranker import BaseReranker, RerankCandidate, RerankResult, RerankerFactory


class CoreRerankerError(ValueError):
    """Raised when reranker input is invalid."""


class CoreReranker:
    """Rerank fused retrieval results while preserving availability on backend failure."""

    def __init__(self, settings: Settings, reranker: BaseReranker | None = None) -> None:
        self.settings = settings
        self.reranker = reranker or RerankerFactory.create(settings)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        """Return reranked retrieval results, or fusion order with fallback metadata."""
        if not isinstance(query, str) or not query.strip():
            raise CoreRerankerError("query must be a non-empty string")
        _validate_candidates(candidates)
        limit = len(candidates) if top_k is None else top_k
        if limit < 0:
            raise CoreRerankerError("top_k must be non-negative")
        if not candidates or limit == 0:
            _record_trace(trace, "core_reranker.rerank", {"candidate_count": len(candidates), "result_count": 0})
            return []

        selected = candidates[:limit]
        try:
            reranked = self.reranker.rerank(
                query.strip(),
                [_to_candidate(candidate) for candidate in selected],
                trace=trace,
            )
            _validate_backend_results(reranked)
            results = _merge_rerank_results(selected, reranked)
            _record_trace(
                trace,
                "core_reranker.rerank",
                {"candidate_count": len(selected), "result_count": len(results), "fallback": False},
            )
            return results
        except Exception as exc:
            results = [_fallback_result(candidate, str(exc)) for candidate in selected]
            _record_trace(
                trace,
                "core_reranker.fallback",
                {"candidate_count": len(selected), "result_count": len(results), "error": str(exc)},
            )
            return results


def _to_candidate(result: RetrievalResult) -> RerankCandidate:
    return RerankCandidate(
        id=result.chunk_id,
        text=result.text,
        score=float(result.score),
        metadata=copy.deepcopy(result.metadata),
    )


def _merge_rerank_results(
    original: list[RetrievalResult],
    reranked: list[RerankResult],
) -> list[RetrievalResult]:
    original_by_id = {candidate.chunk_id: candidate for candidate in original}
    seen: set[str] = set()
    merged: list[RetrievalResult] = []
    for item in sorted(reranked, key=lambda result: result.rank):
        if item.id in seen:
            raise CoreRerankerError("reranker returned duplicate candidate ids")
        if item.id not in original_by_id:
            raise CoreRerankerError(f"reranker returned unknown candidate id: {item.id}")
        seen.add(item.id)
        original_result = original_by_id[item.id]
        metadata = copy.deepcopy(item.metadata or original_result.metadata)
        metadata["rerank"] = {"fallback": False, "rank": item.rank, "score": float(item.score)}
        merged.append(
            RetrievalResult(
                chunk_id=item.id,
                score=float(item.score),
                text=item.text or original_result.text,
                metadata=metadata,
            )
        )
    for candidate in original:
        if candidate.chunk_id not in seen:
            metadata = copy.deepcopy(candidate.metadata)
            metadata["rerank"] = {"fallback": False, "rank": len(merged) + 1, "score": float(candidate.score)}
            merged.append(
                RetrievalResult(
                    chunk_id=candidate.chunk_id,
                    score=float(candidate.score),
                    text=candidate.text,
                    metadata=metadata,
                )
            )
    return merged


def _fallback_result(candidate: RetrievalResult, reason: str) -> RetrievalResult:
    metadata = copy.deepcopy(candidate.metadata)
    metadata["rerank"] = {"fallback": True, "reason": reason}
    return RetrievalResult(
        chunk_id=candidate.chunk_id,
        score=float(candidate.score),
        text=candidate.text,
        metadata=metadata,
    )


def _validate_candidates(candidates: list[RetrievalResult]) -> None:
    if not isinstance(candidates, list):
        raise CoreRerankerError("candidates must be a list")
    for candidate in candidates:
        if not isinstance(candidate, RetrievalResult):
            raise CoreRerankerError("candidates must contain RetrievalResult objects")


def _validate_backend_results(results: list[RerankResult]) -> None:
    if not isinstance(results, list):
        raise CoreRerankerError("reranker backend must return a list")
    for result in results:
        if not isinstance(result, RerankResult):
            raise CoreRerankerError("reranker backend must return RerankResult objects")


def _record_trace(trace: Any | None, name: str, data: dict[str, Any]) -> None:
    if hasattr(trace, "record_stage"):
        trace.record_stage(name, data)
