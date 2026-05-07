"""Tests for BM25 index build, query, and persistence."""

from __future__ import annotations

import math

import pytest

from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from ingestion.embedding import SparseEncoder, SparseEncodingResult
from ingestion.storage import BM25Hit, BM25Indexer, BM25IndexerError


def chunks() -> list[Chunk]:
    return [
        Chunk("a", "alpha beta beta", {"source_path": "docs/a.md"}, 0, 15, "doc"),
        Chunk("b", "beta gamma gamma gamma", {"source_path": "docs/a.md"}, 16, 38, "doc"),
        Chunk("c", "delta alpha", {"source_path": "docs/b.md"}, 0, 11, "doc2"),
    ]


def build_index(tmp_path) -> BM25Indexer:
    sparse = SparseEncoder().encode(chunks())
    indexer = BM25Indexer(index_dir=tmp_path / "bm25")
    indexer.build(sparse)
    return indexer


def test_build_creates_expected_inverted_index_and_idf(tmp_path) -> None:
    indexer = build_index(tmp_path)

    assert indexer.total_chunks == 3
    assert indexer.average_doc_length == pytest.approx(3.0)
    assert set(indexer.inverted_index) == {"alpha", "beta", "delta", "gamma"}
    assert indexer.inverted_index["alpha"]["idf"] == pytest.approx(math.log((3 - 2 + 0.5) / (2 + 0.5)))
    assert indexer.inverted_index["gamma"]["postings"] == [{"chunk_id": "b", "tf": 3.0, "doc_length": 4}]


def test_query_returns_stable_top_hits(tmp_path) -> None:
    indexer = build_index(tmp_path)

    hits = indexer.query("gamma beta", top_k=2)

    assert [hit.chunk_id for hit in hits] == ["b", "a"]
    assert all(isinstance(hit, BM25Hit) for hit in hits)
    assert hits[0].text == "beta gamma gamma gamma"
    assert hits[0].metadata["source_path"] == "docs/a.md"


def test_query_accepts_term_list_and_respects_top_k(tmp_path) -> None:
    indexer = build_index(tmp_path)

    hits = indexer.query(["alpha"], top_k=1)

    assert len(hits) == 1
    assert hits[0].chunk_id in {"a", "c"}


def test_save_and_load_roundtrip_preserves_query_results(tmp_path) -> None:
    indexer = build_index(tmp_path)
    path = indexer.save()

    loaded = BM25Indexer.load(tmp_path / "bm25")

    assert path.is_file()
    assert [hit.to_dict() for hit in loaded.query("gamma beta", top_k=3)] == [
        hit.to_dict() for hit in indexer.query("gamma beta", top_k=3)
    ]


def test_add_incrementally_replaces_existing_records(tmp_path) -> None:
    indexer = build_index(tmp_path)
    replacement = SparseEncoder().encode(
        [Chunk("a", "zeta zeta zeta", {"source_path": "docs/a.md"}, 0, 14, "doc")]
    )

    indexer.add(replacement)

    assert indexer.records["a"].text == "zeta zeta zeta"
    assert indexer.query("zeta", top_k=3)[0].chunk_id == "a"
    assert "alpha" in indexer.inverted_index


def test_build_empty_result_clears_index(tmp_path) -> None:
    indexer = build_index(tmp_path)

    indexer.build(SparseEncoder().encode([]))

    assert indexer.query("alpha", top_k=10) == []
    assert indexer.total_chunks == 0


def test_trace_records_build_and_add(tmp_path) -> None:
    trace = TraceContext()
    indexer = BM25Indexer(index_dir=tmp_path / "bm25")

    indexer.build(SparseEncoder().encode(chunks()), trace=trace)
    indexer.add(SparseEncoder().encode([chunks()[0]]), trace=trace)

    assert trace.stages == [
        {"name": "bm25_indexer.build", "data": {"chunk_count": 3, "terms": 4}},
        {"name": "bm25_indexer.add", "data": {"chunk_count": 1, "terms": 4}},
    ]


def test_invalid_query_parameters_have_readable_errors(tmp_path) -> None:
    indexer = build_index(tmp_path)

    with pytest.raises(BM25IndexerError, match="top_k"):
        indexer.query("alpha", top_k=0)
    with pytest.raises(BM25IndexerError, match="keywords"):
        indexer.query(123, top_k=1)  # type: ignore[arg-type]


def test_load_missing_index_has_readable_error(tmp_path) -> None:
    with pytest.raises(BM25IndexerError, match="not found"):
        BM25Indexer.load(tmp_path / "missing")


def test_record_without_sparse_vector_is_rejected(tmp_path) -> None:
    indexer = BM25Indexer(index_dir=tmp_path / "bm25")

    with pytest.raises(BM25IndexerError, match="missing sparse_vector"):
        indexer.build(
            SparseEncodingResult(
                records=[ChunkRecord(id="x", text="text", metadata={"source_path": "docs/x.md"})],
                document_frequency={},
                total_chunks=1,
                average_doc_length=1.0,
            )
        )
