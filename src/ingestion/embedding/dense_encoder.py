"""Dense vector encoding for ingestion chunks."""

from __future__ import annotations

import copy
from typing import Any

from core.settings import Settings
from core.types import Chunk, ChunkRecord
from libs.embedding import BaseEmbedding, EmbeddingFactory


class DenseEncoderError(ValueError):
    """Raised when dense encoding cannot produce valid chunk records."""


class DenseEncoder:
    """Encode chunk text with the configured embedding backend."""

    def __init__(self, settings: Settings, embedding: BaseEmbedding | None = None) -> None:
        self.settings = settings
        self.embedding = embedding or EmbeddingFactory.create(settings)

    def encode(self, chunks: list[Chunk], trace: Any | None = None) -> list[ChunkRecord]:
        """Return one dense-vector ChunkRecord for each input chunk."""
        if not chunks:
            return []
        for chunk in chunks:
            if not isinstance(chunk, Chunk):
                raise DenseEncoderError("DenseEncoder.encode expects a list of Chunk objects")
            if not chunk.text.strip():
                raise DenseEncoderError(f"chunk {chunk.id} text must be non-empty")

        texts = [chunk.text for chunk in chunks]
        vectors = self.embedding.embed(texts, trace=trace)
        _validate_vectors(vectors, expected_count=len(chunks))

        records = [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=copy.deepcopy(chunk.metadata),
                dense_vector=vector,
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        _record_trace(trace, "dense_encoder.encoded", {"chunk_count": len(records), "dimension": len(vectors[0])})
        return records


def _validate_vectors(vectors: list[list[float]], expected_count: int) -> None:
    if not isinstance(vectors, list):
        raise DenseEncoderError("embedding backend must return a list of vectors")
    if len(vectors) != expected_count:
        raise DenseEncoderError(
            f"embedding vector count mismatch: expected {expected_count}, got {len(vectors)}"
        )
    if not vectors:
        return

    dimension = len(vectors[0])
    if dimension == 0:
        raise DenseEncoderError("embedding vectors must not be empty")
    for index, vector in enumerate(vectors):
        if not isinstance(vector, list) or not all(isinstance(value, (int, float)) for value in vector):
            raise DenseEncoderError(f"embedding vector[{index}] must be a list of numbers")
        if len(vector) != dimension:
            raise DenseEncoderError(
                f"embedding vector dimension mismatch at index {index}: expected {dimension}, got {len(vector)}"
            )


def _record_trace(trace: Any | None, name: str, data: dict[str, Any]) -> None:
    if hasattr(trace, "record_stage"):
        trace.record_stage(name, data)
