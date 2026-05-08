"""Retrieval result fusion algorithms."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from core.types import RetrievalResult


class FusionError(ValueError):
    """Raised when fusion input or configuration is invalid."""


@dataclass(frozen=True)
class FusionContribution:
    """Per-route contribution to a fused result."""

    route: str
    rank: int
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize contribution details for debugging and traces."""
        return {"route": self.route, "rank": self.rank, "score": self.score}


class RRFusion:
    """Reciprocal Rank Fusion for dense and sparse retrieval outputs."""

    def __init__(self, k: int = 60) -> None:
        if k <= 0:
            raise FusionError("k must be greater than 0")
        self.k = int(k)

    def fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        """Fuse dense and sparse ranked lists into a deterministic top-k list."""
        if top_k <= 0:
            raise FusionError("top_k must be greater than 0")
        _validate_results("dense_results", dense_results)
        _validate_results("sparse_results", sparse_results)

        candidates: dict[str, dict[str, Any]] = {}
        self._accumulate(candidates, dense_results, route="dense")
        self._accumulate(candidates, sparse_results, route="sparse")

        ordered = sorted(
            candidates.values(),
            key=lambda item: (-item["fusion_score"], item["best_rank"], item["chunk_id"]),
        )[:top_k]
        results = [_to_result(item) for item in ordered]
        _record_trace(
            trace,
            "fusion.rrf",
            {
                "algorithm": "rrf",
                "k": self.k,
                "dense_count": len(dense_results),
                "sparse_count": len(sparse_results),
                "result_count": len(results),
            },
        )
        return results

    def _accumulate(
        self,
        candidates: dict[str, dict[str, Any]],
        results: list[RetrievalResult],
        route: str,
    ) -> None:
        seen_in_route: set[str] = set()
        for rank, result in enumerate(results, start=1):
            if result.chunk_id in seen_in_route:
                continue
            seen_in_route.add(result.chunk_id)
            contribution = 1.0 / (self.k + rank)
            item = candidates.setdefault(
                result.chunk_id,
                {
                    "chunk_id": result.chunk_id,
                    "text": result.text,
                    "metadata": copy.deepcopy(result.metadata),
                    "fusion_score": 0.0,
                    "best_rank": rank,
                    "contributions": [],
                },
            )
            item["fusion_score"] += contribution
            item["best_rank"] = min(item["best_rank"], rank)
            item["contributions"].append(
                FusionContribution(route=route, rank=rank, score=float(result.score)).to_dict()
            )


def _to_result(item: dict[str, Any]) -> RetrievalResult:
    metadata = copy.deepcopy(item["metadata"])
    metadata["fusion"] = {
        "algorithm": "rrf",
        "score": item["fusion_score"],
        "best_rank": item["best_rank"],
        "contributions": item["contributions"],
    }
    return RetrievalResult(
        chunk_id=item["chunk_id"],
        score=float(item["fusion_score"]),
        text=item["text"],
        metadata=metadata,
    )


def _validate_results(name: str, results: list[RetrievalResult]) -> None:
    if not isinstance(results, list):
        raise FusionError(f"{name} must be a list")
    for result in results:
        if not isinstance(result, RetrievalResult):
            raise FusionError(f"{name} must contain RetrievalResult objects")


def _record_trace(trace: Any | None, name: str, data: dict[str, Any]) -> None:
    if hasattr(trace, "record_stage"):
        trace.record_stage(name, data)
