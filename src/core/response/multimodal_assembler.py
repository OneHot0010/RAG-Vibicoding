"""Assemble MCP multimodal content from retrieved chunk metadata."""

from __future__ import annotations

import base64
import mimetypes
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.types import RetrievalResult, normalize_image_refs


@dataclass(frozen=True)
class AssembledImage:
    """Image content and metadata ready for an MCP tool response."""

    image_id: str
    path: str
    mime_type: str
    data: str
    chunk_id: str
    citation_index: int

    def to_content(self) -> dict[str, Any]:
        """Serialize as MCP ImageContent."""
        return {"type": "image", "data": self.data, "mimeType": self.mime_type}

    def to_dict(self) -> dict[str, Any]:
        """Serialize image metadata without duplicating base64 in structuredContent."""
        return {
            "image_id": self.image_id,
            "path": self.path,
            "mime_type": self.mime_type,
            "chunk_id": self.chunk_id,
            "citation_index": self.citation_index,
        }


@dataclass(frozen=True)
class MultimodalAssembly:
    """Result of scanning retrieval hits for images."""

    images: list[AssembledImage]
    skipped: list[dict[str, Any]]

    def to_content(self) -> list[dict[str, Any]]:
        """Return all image content items for MCP content arrays."""
        return [image.to_content() for image in self.images]

    def to_dict(self) -> dict[str, Any]:
        """Return structured assembly metadata."""
        return {
            "images": [image.to_dict() for image in self.images],
            "skipped": self.skipped,
        }


class MultimodalAssembler:
    """Read image files referenced by retrieval results and encode them for MCP."""

    def __init__(self, data_dir: str | Path = "data", max_images: int = 8) -> None:
        if max_images <= 0:
            raise ValueError("max_images must be greater than 0")
        self.data_dir = Path(data_dir)
        self.max_images = max_images

    def assemble(self, retrieval_results: list[RetrievalResult]) -> MultimodalAssembly:
        """Return image content for images referenced by retrieval results."""
        if not isinstance(retrieval_results, list):
            raise ValueError("retrieval_results must be a list")

        images: list[AssembledImage] = []
        skipped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for citation_index, result in enumerate(retrieval_results, start=1):
            if not isinstance(result, RetrievalResult):
                raise ValueError("retrieval_results must contain RetrievalResult objects")
            for image_id in _image_ids(result.metadata):
                if image_id in seen:
                    continue
                seen.add(image_id)
                resolved = self._resolve_image(image_id, result.metadata)
                if resolved is None:
                    skipped.append({"image_id": image_id, "chunk_id": result.chunk_id, "reason": "not_found"})
                    continue
                path = resolved
                if not path.is_file():
                    skipped.append({"image_id": image_id, "chunk_id": result.chunk_id, "reason": "file_missing", "path": str(path)})
                    continue
                try:
                    data = base64.b64encode(path.read_bytes()).decode("ascii")
                except OSError:
                    skipped.append({"image_id": image_id, "chunk_id": result.chunk_id, "reason": "read_failed", "path": str(path)})
                    continue
                images.append(
                    AssembledImage(
                        image_id=image_id,
                        path=str(path),
                        mime_type=_mime_type(path),
                        data=data,
                        chunk_id=result.chunk_id,
                        citation_index=citation_index,
                    )
                )
                if len(images) >= self.max_images:
                    return MultimodalAssembly(images=images, skipped=skipped)
        return MultimodalAssembly(images=images, skipped=skipped)

    def _resolve_image(self, image_id: str, metadata: dict[str, Any]) -> Path | None:
        direct = _direct_image_path(image_id, metadata)
        if direct is not None:
            return direct
        indexed = self._indexed_image_path(image_id)
        if indexed is not None:
            return indexed
        collection = metadata.get("collection")
        if isinstance(collection, str) and collection.strip():
            for suffix in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
                candidate = self.data_dir / "images" / collection.strip() / f"{image_id}{suffix}"
                if candidate.is_file():
                    return candidate
        return None

    def _indexed_image_path(self, image_id: str) -> Path | None:
        db_path = self.data_dir / "db" / "image_index.db"
        if not db_path.is_file():
            return None
        try:
            with sqlite3.connect(db_path) as connection:
                row = connection.execute(
                    "SELECT file_path FROM image_index WHERE image_id = ?",
                    (image_id,),
                ).fetchone()
        except sqlite3.Error:
            return None
        if not row:
            return None
        return Path(str(row[0]))


def _image_ids(metadata: dict[str, Any]) -> list[str]:
    raw_refs = metadata.get("image_refs")
    if isinstance(raw_refs, list):
        refs = [str(item) for item in raw_refs if isinstance(item, str) and item.strip()]
        if refs:
            return refs
    images = _normalized_images(metadata)
    return [image["id"] for image in images if image.get("id")]


def _direct_image_path(image_id: str, metadata: dict[str, Any]) -> Path | None:
    for image in _normalized_images(metadata):
        if image.get("id") != image_id:
            continue
        path = image.get("path")
        if isinstance(path, str) and path.strip():
            return Path(path)
    return None


def _normalized_images(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    images = metadata.get("images")
    if not isinstance(images, list):
        return []
    try:
        return normalize_image_refs(images)
    except ValueError:
        return []


def _mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "image/png"
