"""Core data contracts shared by ingestion, retrieval, and MCP layers."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[IMAGE:\s*([A-Za-z0-9_.:-]+)\]")


class CoreTypeError(ValueError):
    """Raised when a core data contract is invalid."""


@dataclass(frozen=True)
class ImageRef:
    """Normalized image reference stored inside document or chunk metadata."""

    id: str
    path: str
    page: int | None = None
    text_offset: int = 0
    text_length: int = 0
    position: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty("ImageRef.id", self.id)
        _require_non_empty("ImageRef.path", self.path)
        if self.page is not None and self.page < 0:
            raise CoreTypeError("ImageRef.page must be non-negative when provided")
        if self.text_offset < 0:
            raise CoreTypeError("ImageRef.text_offset must be non-negative")
        if self.text_length < 0:
            raise CoreTypeError("ImageRef.text_length must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this image reference using stable field names."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImageRef":
        """Create an image reference from a metadata mapping."""
        return cls(
            id=str(data.get("id") or ""),
            path=str(data.get("path") or ""),
            page=data.get("page"),
            text_offset=int(data.get("text_offset") or 0),
            text_length=int(data.get("text_length") or 0),
            position=dict(data.get("position") or {}),
        )


@dataclass(frozen=True)
class Document:
    """Loaded document text plus source metadata."""

    id: str
    text: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        _require_non_empty("Document.id", self.id)
        _require_text("Document.text", self.text)
        _validate_metadata("Document.metadata", self.metadata)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this document into JSON-friendly primitives."""
        return {"id": self.id, "text": self.text, "metadata": _serialize_metadata(self.metadata)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Document":
        """Create a document from a serialized mapping."""
        return cls(
            id=str(data.get("id") or ""),
            text=str(data.get("text") or ""),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class Chunk:
    """A positioned text chunk derived from a source document."""

    id: str
    text: str
    metadata: dict[str, Any]
    start_offset: int
    end_offset: int
    source_ref: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty("Chunk.id", self.id)
        _require_text("Chunk.text", self.text)
        _validate_metadata("Chunk.metadata", self.metadata)
        if self.start_offset < 0:
            raise CoreTypeError("Chunk.start_offset must be non-negative")
        if self.end_offset < self.start_offset:
            raise CoreTypeError("Chunk.end_offset must be greater than or equal to start_offset")
        if self.source_ref is not None:
            _require_non_empty("Chunk.source_ref", self.source_ref)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this chunk into JSON-friendly primitives."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": _serialize_metadata(self.metadata),
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Chunk":
        """Create a chunk from a serialized mapping."""
        return cls(
            id=str(data.get("id") or ""),
            text=str(data.get("text") or ""),
            metadata=dict(data.get("metadata") or {}),
            start_offset=int(data.get("start_offset") or 0),
            end_offset=int(data.get("end_offset") or 0),
            source_ref=data.get("source_ref"),
        )


@dataclass(frozen=True)
class ChunkRecord:
    """Storage and retrieval carrier for chunk text, metadata, and vectors."""

    id: str
    text: str
    metadata: dict[str, Any]
    dense_vector: list[float] | None = None
    sparse_vector: dict[str, float] | None = None

    def __post_init__(self) -> None:
        _require_non_empty("ChunkRecord.id", self.id)
        _require_text("ChunkRecord.text", self.text)
        _validate_metadata("ChunkRecord.metadata", self.metadata)
        if self.dense_vector is not None:
            _validate_dense_vector(self.dense_vector)
        if self.sparse_vector is not None:
            _validate_sparse_vector(self.sparse_vector)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this chunk record into JSON-friendly primitives."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": _serialize_metadata(self.metadata),
            "dense_vector": self.dense_vector,
            "sparse_vector": self.sparse_vector,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChunkRecord":
        """Create a chunk record from a serialized mapping."""
        sparse_vector = data.get("sparse_vector")
        return cls(
            id=str(data.get("id") or ""),
            text=str(data.get("text") or ""),
            metadata=dict(data.get("metadata") or {}),
            dense_vector=list(data["dense_vector"]) if data.get("dense_vector") is not None else None,
            sparse_vector=dict(sparse_vector) if sparse_vector is not None else None,
        )


def image_placeholder(image_id: str) -> str:
    """Return the canonical image placeholder for document text."""
    _require_non_empty("image_id", image_id)
    return f"[IMAGE: {image_id}]"


def extract_image_placeholders(text: str) -> list[str]:
    """Return image ids referenced by canonical placeholders in text order."""
    if not isinstance(text, str):
        raise CoreTypeError("text must be a string")
    return [match.group(1) for match in IMAGE_PLACEHOLDER_PATTERN.finditer(text)]


def normalize_image_refs(images: list[ImageRef | Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize image refs into the metadata.images list-of-dicts contract."""
    if not images:
        return []
    return [image.to_dict() if isinstance(image, ImageRef) else ImageRef.from_dict(image).to_dict() for image in images]


def _serialize_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    serialized = dict(metadata)
    if "images" in serialized:
        images = serialized["images"]
        if not isinstance(images, list):
            raise CoreTypeError("metadata.images must be a list")
        serialized["images"] = normalize_image_refs(images)
    return serialized


def _validate_metadata(field_name: str, metadata: Mapping[str, Any]) -> None:
    if not isinstance(metadata, Mapping):
        raise CoreTypeError(f"{field_name} must be a mapping")
    source_path = metadata.get("source_path")
    if not isinstance(source_path, str) or not source_path:
        raise CoreTypeError(f"{field_name}.source_path is required")
    if "images" in metadata:
        images = metadata["images"]
        if not isinstance(images, list):
            raise CoreTypeError(f"{field_name}.images must be a list")
        normalize_image_refs(images)


def _require_non_empty(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CoreTypeError(f"{field_name} must be a non-empty string")


def _require_text(field_name: str, value: str) -> None:
    if not isinstance(value, str):
        raise CoreTypeError(f"{field_name} must be a string")


def _validate_dense_vector(vector: list[float]) -> None:
    if not isinstance(vector, list) or not all(isinstance(item, (int, float)) for item in vector):
        raise CoreTypeError("ChunkRecord.dense_vector must be a list of numbers")


def _validate_sparse_vector(vector: dict[str, float]) -> None:
    if not isinstance(vector, dict):
        raise CoreTypeError("ChunkRecord.sparse_vector must be a mapping")
    if not all(isinstance(term, str) and isinstance(score, (int, float)) for term, score in vector.items()):
        raise CoreTypeError("ChunkRecord.sparse_vector must map string terms to numeric scores")
