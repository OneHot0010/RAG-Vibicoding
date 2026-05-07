"""Sparse term encoding for BM25-style indexing."""

from __future__ import annotations

import copy
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from core.types import Chunk, ChunkRecord


class SparseEncoderError(ValueError):
    """Raised when sparse encoding input is invalid."""


@dataclass(frozen=True)
class SparseEncodingResult:
    """Sparse encoding output consumed by the future BM25 indexer."""

    records: list[ChunkRecord]
    document_frequency: dict[str, int]
    total_chunks: int
    average_doc_length: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize sparse encoding statistics with stable field names."""
        return {
            "records": [record.to_dict() for record in self.records],
            "document_frequency": dict(self.document_frequency),
            "total_chunks": self.total_chunks,
            "average_doc_length": self.average_doc_length,
        }


class SparseEncoder:
    """Encode chunks into term-frequency sparse vectors and corpus statistics."""

    def __init__(self, lowercase: bool = True, min_token_length: int = 2) -> None:
        if min_token_length <= 0:
            raise SparseEncoderError("min_token_length must be greater than 0")
        self.lowercase = lowercase
        self.min_token_length = min_token_length

    def encode(self, chunks: list[Chunk], trace: Any | None = None) -> SparseEncodingResult:
        """Return sparse vectors plus BM25-ready document frequency statistics."""
        if not chunks:
            result = SparseEncodingResult([], {}, 0, 0.0)
            _record_trace(trace, "sparse_encoder.encoded", {"chunk_count": 0, "vocabulary_size": 0})
            return result

        records: list[ChunkRecord] = []
        document_frequency: Counter[str] = Counter()
        total_doc_length = 0

        for chunk in chunks:
            if not isinstance(chunk, Chunk):
                raise SparseEncoderError("SparseEncoder.encode expects a list of Chunk objects")
            tokens = self.tokenize(chunk.text)
            if not tokens:
                raise SparseEncoderError(f"chunk {chunk.id} text produced no sparse tokens")

            term_counts = Counter(tokens)
            document_frequency.update(term_counts.keys())
            total_doc_length += sum(term_counts.values())
            records.append(
                ChunkRecord(
                    id=chunk.id,
                    text=chunk.text,
                    metadata={
                        **copy.deepcopy(chunk.metadata),
                        "doc_length": sum(term_counts.values()),
                    },
                    sparse_vector={term: float(count) for term, count in sorted(term_counts.items())},
                )
            )

        result = SparseEncodingResult(
            records=records,
            document_frequency=dict(sorted(document_frequency.items())),
            total_chunks=len(records),
            average_doc_length=total_doc_length / len(records),
        )
        _record_trace(
            trace,
            "sparse_encoder.encoded",
            {"chunk_count": len(records), "vocabulary_size": len(result.document_frequency)},
        )
        return result

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text for sparse indexing."""
        if not isinstance(text, str):
            raise SparseEncoderError("text must be a string")
        source = text.lower() if self.lowercase else text
        return [
            token
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", source)
            if len(token) >= self.min_token_length and token not in _STOP_WORDS
        ]


def _record_trace(trace: Any | None, name: str, data: dict[str, Any]) -> None:
    if hasattr(trace, "record_stage"):
        trace.record_stage(name, data)


_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
