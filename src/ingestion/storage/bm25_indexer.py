"""BM25 index construction, persistence, and querying."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.types import ChunkRecord
from ingestion.embedding.sparse_encoder import SparseEncodingResult


DEFAULT_BM25_DIR = Path("data/db/bm25")


class BM25IndexerError(ValueError):
    """Raised when BM25 index input or query parameters are invalid."""


@dataclass(frozen=True)
class BM25Hit:
    """A query hit returned by BM25 search."""

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize hit for callers and tests."""
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata,
        }


class BM25Indexer:
    """Build and persist a BM25 inverted index from sparse encoding output."""

    index_filename = "index.json"

    def __init__(self, index_dir: str | Path = DEFAULT_BM25_DIR, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise BM25IndexerError("k1 must be greater than 0")
        if not 0 <= b <= 1:
            raise BM25IndexerError("b must be between 0 and 1")
        self.index_dir = Path(index_dir)
        self.k1 = float(k1)
        self.b = float(b)
        self.records: dict[str, ChunkRecord] = {}
        self.inverted_index: dict[str, dict[str, Any]] = {}
        self.total_chunks = 0
        self.average_doc_length = 0.0

    def build(self, sparse_result: SparseEncodingResult, trace: Any | None = None) -> None:
        """Replace the index with the given sparse encoding result."""
        self.records = {record.id: record for record in sparse_result.records}
        self._rebuild_index()
        _record_trace(trace, "bm25_indexer.build", {"chunk_count": self.total_chunks, "terms": len(self.inverted_index)})

    def add(self, sparse_result: SparseEncodingResult, trace: Any | None = None) -> None:
        """Incrementally add or replace records, then recalculate index statistics."""
        for record in sparse_result.records:
            self.records[record.id] = record
        self._rebuild_index()
        _record_trace(trace, "bm25_indexer.add", {"chunk_count": len(sparse_result.records), "terms": len(self.inverted_index)})

    def query(self, keywords: str | list[str], top_k: int = 10) -> list[BM25Hit]:
        """Query the BM25 index and return top hits by score."""
        if top_k <= 0:
            raise BM25IndexerError("top_k must be greater than 0")
        terms = _normalize_query_terms(keywords)
        if not terms or not self.inverted_index:
            return []

        scores: dict[str, float] = {}
        for term in terms:
            entry = self.inverted_index.get(term)
            if not entry:
                continue
            idf = float(entry["idf"])
            for posting in entry["postings"]:
                chunk_id = posting["chunk_id"]
                tf = float(posting["tf"])
                doc_length = float(posting["doc_length"])
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / max(self.average_doc_length, 1e-9))
                score = idf * (tf * (self.k1 + 1)) / denominator
                scores[chunk_id] = scores.get(chunk_id, 0.0) + score

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [
            BM25Hit(
                chunk_id=chunk_id,
                score=score,
                text=self.records[chunk_id].text,
                metadata=self.records[chunk_id].metadata,
            )
            for chunk_id, score in ordered
        ]

    def save(self) -> Path:
        """Persist the current index to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        path = self.index_dir / self.index_filename
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, index_dir: str | Path = DEFAULT_BM25_DIR) -> "BM25Indexer":
        """Load a BM25 index from disk."""
        indexer = cls(index_dir=index_dir)
        path = indexer.index_dir / indexer.index_filename
        if not path.exists():
            raise BM25IndexerError(f"BM25 index not found: {path}")
        decoded = json.loads(path.read_text(encoding="utf-8"))
        indexer.k1 = float(decoded["k1"])
        indexer.b = float(decoded["b"])
        indexer.records = {
            item["id"]: ChunkRecord.from_dict(item)
            for item in decoded.get("records", [])
        }
        indexer.inverted_index = decoded.get("inverted_index", {})
        indexer.total_chunks = int(decoded.get("total_chunks", 0))
        indexer.average_doc_length = float(decoded.get("average_doc_length", 0.0))
        return indexer

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full index."""
        return {
            "k1": self.k1,
            "b": self.b,
            "total_chunks": self.total_chunks,
            "average_doc_length": self.average_doc_length,
            "records": [record.to_dict() for record in self.records.values()],
            "inverted_index": self.inverted_index,
        }

    def _rebuild_index(self) -> None:
        self.total_chunks = len(self.records)
        if self.total_chunks == 0:
            self.average_doc_length = 0.0
            self.inverted_index = {}
            return

        doc_lengths = {
            record.id: _doc_length(record)
            for record in self.records.values()
        }
        self.average_doc_length = sum(doc_lengths.values()) / self.total_chunks

        document_frequency: dict[str, int] = {}
        for record in self.records.values():
            sparse_vector = _sparse_vector(record)
            for term in sparse_vector:
                document_frequency[term] = document_frequency.get(term, 0) + 1

        index: dict[str, dict[str, Any]] = {}
        for term, df in sorted(document_frequency.items()):
            postings = []
            for record in self.records.values():
                sparse_vector = _sparse_vector(record)
                if term not in sparse_vector:
                    continue
                postings.append(
                    {
                        "chunk_id": record.id,
                        "tf": float(sparse_vector[term]),
                        "doc_length": doc_lengths[record.id],
                    }
                )
            index[term] = {
                "idf": _idf(self.total_chunks, df),
                "postings": sorted(postings, key=lambda item: item["chunk_id"]),
            }
        self.inverted_index = index


def _idf(total_chunks: int, document_frequency: int) -> float:
    return math.log((total_chunks - document_frequency + 0.5) / (document_frequency + 0.5))


def _sparse_vector(record: ChunkRecord) -> dict[str, float]:
    if not record.sparse_vector:
        raise BM25IndexerError(f"record {record.id} missing sparse_vector")
    return record.sparse_vector


def _doc_length(record: ChunkRecord) -> int:
    if isinstance(record.metadata.get("doc_length"), (int, float)):
        return int(record.metadata["doc_length"])
    return int(sum(_sparse_vector(record).values()))


def _normalize_query_terms(keywords: str | list[str]) -> list[str]:
    if isinstance(keywords, str):
        terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", keywords.lower())
    elif isinstance(keywords, list) and all(isinstance(term, str) for term in keywords):
        terms = [term.lower() for term in keywords]
    else:
        raise BM25IndexerError("keywords must be a string or list[str]")
    return [term for term in terms if term]


def _record_trace(trace: Any | None, name: str, data: dict[str, Any]) -> None:
    if hasattr(trace, "record_stage"):
        trace.record_stage(name, data)
