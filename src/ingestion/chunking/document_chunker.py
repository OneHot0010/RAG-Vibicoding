"""Adapt text splitters into Document -> Chunk conversion."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from core.settings import Settings
from core.types import Chunk, Document, extract_image_placeholders, normalize_image_refs
from libs.splitter import BaseSplitter, SplitterFactory


class DocumentChunkerError(ValueError):
    """Raised when a document cannot be converted into chunks."""


class DocumentChunker:
    """Split a normalized Document into business-level Chunk objects."""

    def __init__(self, settings: Settings, splitter: BaseSplitter | None = None) -> None:
        self.settings = settings
        self.splitter = splitter or SplitterFactory.create(settings)

    def split_document(self, document: Document, trace: Any | None = None) -> list[Chunk]:
        """Split a document and enrich text chunks with stable ids and metadata."""
        if not isinstance(document, Document):
            raise DocumentChunkerError("document must be a core.types.Document")

        chunk_texts = self.splitter.split_text(document.text, trace=trace)
        chunks: list[Chunk] = []
        search_offset = 0
        for index, chunk_text in enumerate(chunk_texts):
            if not chunk_text:
                continue
            start_offset = _find_offset(document.text, chunk_text, search_offset)
            end_offset = start_offset + len(chunk_text)
            search_offset = max(start_offset + 1, end_offset)
            chunks.append(
                Chunk(
                    id=self._generate_chunk_id(document.id, index, chunk_text),
                    text=chunk_text,
                    metadata=self._inherit_metadata(document, index, chunk_text),
                    start_offset=start_offset,
                    end_offset=end_offset,
                    source_ref=document.id,
                )
            )
        return chunks

    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        """Generate a deterministic chunk id for a document chunk."""
        if index < 0:
            raise DocumentChunkerError("chunk index must be non-negative")
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{content_hash}"

    def _inherit_metadata(self, document: Document, chunk_index: int, chunk_text: str) -> dict[str, Any]:
        """Copy document metadata and attach chunk-local image references."""
        metadata = copy.deepcopy(document.metadata)
        document_images = normalize_image_refs(metadata.pop("images", []))
        metadata["chunk_index"] = chunk_index

        image_refs = extract_image_placeholders(chunk_text)
        if image_refs:
            images_by_id = {image["id"]: image for image in document_images}
            local_images = [images_by_id[image_id] for image_id in image_refs if image_id in images_by_id]
            metadata["image_refs"] = [image["id"] for image in local_images]
            metadata["images"] = local_images
        return metadata


def _find_offset(text: str, chunk_text: str, start: int) -> int:
    offset = text.find(chunk_text, start)
    if offset >= 0:
        return offset
    offset = text.find(chunk_text)
    if offset >= 0:
        return offset
    return start
