"""Tests for sparse BM25-ready encoding."""

from __future__ import annotations

import pytest

from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from ingestion.embedding import SparseEncoder, SparseEncoderError, SparseEncodingResult


def chunks() -> list[Chunk]:
    return [
        Chunk("chunk-1", "Alpha beta beta retrieval.", {"source_path": "docs/a.md"}, 0, 27, "doc-1"),
        Chunk("chunk-2", "Beta gamma retrieval retrieval.", {"source_path": "docs/a.md"}, 28, 60, "doc-1"),
    ]


def test_sparse_encoder_outputs_records_and_corpus_statistics() -> None:
    result = SparseEncoder().encode(chunks())

    assert isinstance(result, SparseEncodingResult)
    assert result.total_chunks == 2
    assert result.average_doc_length == 4.0
    assert result.document_frequency == {
        "alpha": 1,
        "beta": 2,
        "gamma": 1,
        "retrieval": 2,
    }
    assert result.records == [
        ChunkRecord(
            id="chunk-1",
            text="Alpha beta beta retrieval.",
            metadata={"source_path": "docs/a.md", "doc_length": 4},
            sparse_vector={"alpha": 1.0, "beta": 2.0, "retrieval": 1.0},
        ),
        ChunkRecord(
            id="chunk-2",
            text="Beta gamma retrieval retrieval.",
            metadata={"source_path": "docs/a.md", "doc_length": 4},
            sparse_vector={"beta": 1.0, "gamma": 1.0, "retrieval": 2.0},
        ),
    ]


def test_sparse_encoder_returns_empty_result_for_empty_chunks() -> None:
    result = SparseEncoder().encode([])

    assert result == SparseEncodingResult([], {}, 0, 0.0)


def test_sparse_encoder_result_is_serializable() -> None:
    result = SparseEncoder().encode(chunks())

    serialized = result.to_dict()

    assert serialized["total_chunks"] == 2
    assert serialized["records"][0]["sparse_vector"] == {"alpha": 1.0, "beta": 2.0, "retrieval": 1.0}


def test_tokenizer_lowercases_and_filters_stop_words() -> None:
    encoder = SparseEncoder()

    assert encoder.tokenize("The RAG-system is in production") == ["rag-system", "production"]


def test_tokenizer_can_preserve_case() -> None:
    encoder = SparseEncoder(lowercase=False)

    assert encoder.tokenize("RAG rag") == ["RAG", "rag"]


def test_min_token_length_is_configurable() -> None:
    encoder = SparseEncoder(min_token_length=1)

    assert encoder.tokenize("x yz") == ["x", "yz"]


def test_trace_records_sparse_encoding_stage() -> None:
    trace = TraceContext()

    SparseEncoder().encode(chunks(), trace=trace)

    assert trace.stages == [
        {"name": "sparse_encoder.encoded", "data": {"chunk_count": 2, "vocabulary_size": 4}}
    ]


def test_non_chunk_input_is_rejected() -> None:
    with pytest.raises(SparseEncoderError, match="Chunk objects"):
        SparseEncoder().encode(["not chunk"])  # type: ignore[list-item]


def test_empty_text_has_clear_behavior() -> None:
    empty = Chunk("empty", "   ", {"source_path": "docs/a.md"}, 0, 3)

    with pytest.raises(SparseEncoderError, match="produced no sparse tokens"):
        SparseEncoder().encode([empty])


def test_invalid_min_token_length_is_rejected() -> None:
    with pytest.raises(SparseEncoderError, match="min_token_length"):
        SparseEncoder(min_token_length=0)


def test_non_string_tokenize_input_is_rejected() -> None:
    with pytest.raises(SparseEncoderError, match="text must be a string"):
        SparseEncoder().tokenize(None)  # type: ignore[arg-type]
