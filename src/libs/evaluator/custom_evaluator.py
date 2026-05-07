"""Lightweight deterministic retrieval metrics."""

from __future__ import annotations

from typing import Any

from libs.evaluator.base_evaluator import BaseEvaluator, EvaluationCase, EvaluationResult


class CustomEvaluator(BaseEvaluator):
    """Compute simple retrieval metrics without external dependencies."""

    def evaluate(self, case: EvaluationCase, trace: Any | None = None) -> EvaluationResult:
        golden = set(case.golden_ids)
        retrieved = case.retrieved_ids
        first_hit_rank = _first_hit_rank(retrieved, golden)
        hit_rate = 1.0 if first_hit_rank is not None else 0.0
        mrr = 1.0 / first_hit_rank if first_hit_rank is not None else 0.0

        return EvaluationResult(
            metrics={
                "hit_rate": hit_rate,
                "mrr": mrr,
                "retrieved_count": float(len(retrieved)),
                "golden_count": float(len(golden)),
            },
            details={
                "query": case.query,
                "first_hit_rank": first_hit_rank,
                "matched_ids": [item_id for item_id in retrieved if item_id in golden],
            },
        )


def _first_hit_rank(retrieved_ids: list[str], golden_ids: set[str]) -> int | None:
    if not golden_ids:
        return None
    for index, item_id in enumerate(retrieved_ids, start=1):
        if item_id in golden_ids:
            return index
    return None
