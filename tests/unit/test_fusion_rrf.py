"""Tests for reciprocal rank fusion."""

from __future__ import annotations

import pytest

from core.query_engine import FusionContribution, FusionError, RRFusion
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult


def result(chunk_id: str, score: float = 1.0, text: str | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text or f"{chunk_id} text",
        metadata={"source_path": f"docs/{chunk_id}.md"},
    )


def test_rrf_fuses_dense_and_sparse_rankings_by_formula() -> None:
    fusion = RRFusion(k=60)

    fused = fusion.fuse(
        dense_results=[result("a", 0.9), result("b", 0.8)],
        sparse_results=[result("b", 3.0), result("c", 2.0)],
        top_k=3,
    )

    assert [item.chunk_id for item in fused] == ["b", "a", "c"]
    assert fused[0].score == pytest.approx((1 / 62) + (1 / 61))
    assert fused[1].score == pytest.approx(1 / 61)
    assert fused[2].score == pytest.approx(1 / 62)
    assert fused[0].metadata["fusion"]["contributions"] == [
        {"route": "dense", "rank": 2, "score": 0.8},
        {"route": "sparse", "rank": 1, "score": 3.0},
    ]


def test_top_k_limits_results_and_preserves_result_content() -> None:
    fused = RRFusion(k=10).fuse(
        dense_results=[result("a", text="dense a"), result("b")],
        sparse_results=[result("c")],
        top_k=1,
    )

    assert len(fused) == 1
    assert fused[0].chunk_id == "a"
    assert fused[0].text == "dense a"
    assert fused[0].metadata["source_path"] == "docs/a.md"


def test_tied_scores_are_ordered_by_best_rank_then_chunk_id() -> None:
    fused = RRFusion(k=60).fuse(
        dense_results=[result("b"), result("c")],
        sparse_results=[result("a"), result("c")],
        top_k=3,
    )

    assert [item.chunk_id for item in fused] == ["c", "a", "b"]


def test_duplicate_chunk_within_same_route_counts_once() -> None:
    fused = RRFusion(k=10).fuse(
        dense_results=[result("a", 0.9), result("a", 0.1)],
        sparse_results=[],
        top_k=3,
    )

    assert [item.chunk_id for item in fused] == ["a"]
    assert fused[0].score == pytest.approx(1 / 11)
    assert fused[0].metadata["fusion"]["contributions"] == [
        {"route": "dense", "rank": 1, "score": 0.9}
    ]


def test_custom_k_changes_score() -> None:
    low_k_score = RRFusion(k=10).fuse([result("a")], [], top_k=1)[0].score
    high_k_score = RRFusion(k=100).fuse([result("a")], [], top_k=1)[0].score

    assert low_k_score == pytest.approx(1 / 11)
    assert high_k_score == pytest.approx(1 / 101)
    assert low_k_score > high_k_score


def test_empty_inputs_return_empty_list() -> None:
    assert RRFusion().fuse([], [], top_k=5) == []


def test_trace_records_fusion_summary() -> None:
    trace = TraceContext()

    RRFusion(k=20).fuse([result("a")], [result("b")], top_k=2, trace=trace)

    assert trace.stages == [
        {
            "name": "fusion.rrf",
            "data": {
                "algorithm": "rrf",
                "k": 20,
                "dense_count": 1,
                "sparse_count": 1,
                "result_count": 2,
            },
        }
    ]


def test_contribution_serializes_stable_shape() -> None:
    assert FusionContribution(route="dense", rank=1, score=0.5).to_dict() == {
        "route": "dense",
        "rank": 1,
        "score": 0.5,
    }


def test_invalid_inputs_have_readable_errors() -> None:
    with pytest.raises(FusionError, match="k"):
        RRFusion(k=0)
    with pytest.raises(FusionError, match="top_k"):
        RRFusion().fuse([], [], top_k=0)
    with pytest.raises(FusionError, match="dense_results"):
        RRFusion().fuse(["bad"], [], top_k=1)  # type: ignore[list-item]
    with pytest.raises(FusionError, match="sparse_results"):
        RRFusion().fuse([], ["bad"], top_k=1)  # type: ignore[list-item]
