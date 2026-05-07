"""Metadata enrichment transform for chunk title, summary, and tags."""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from typing import Any

from core.settings import Settings
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm import BaseLLM, ChatMessage, LLMFactory


class MetadataEnricher(BaseTransform):
    """Add retrieval-friendly title, summary, and tags to chunks."""

    def __init__(self, settings: Settings, llm: BaseLLM | None = None) -> None:
        self.settings = settings
        self.llm = llm

    def transform(self, chunks: list[Chunk], trace: Any | None = None) -> list[Chunk]:
        """Enrich chunks independently so failures do not block ingestion."""
        enriched: list[Chunk] = []
        for chunk in chunks:
            try:
                enriched.append(self._transform_one(chunk, trace))
            except Exception as exc:
                metadata = copy.deepcopy(chunk.metadata)
                metadata.update(_rule_metadata(chunk.text))
                metadata["metadata_enriched_by"] = "rule"
                metadata["metadata_enrichment_error"] = str(exc)
                enriched.append(_copy_chunk(chunk, metadata))
        return enriched

    def _transform_one(self, chunk: Chunk, trace: Any | None) -> Chunk:
        metadata = copy.deepcopy(chunk.metadata)
        rule_metadata = _rule_metadata(chunk.text)
        metadata.update(rule_metadata)
        metadata["metadata_enriched_by"] = "rule"

        if not self._use_llm():
            _record_trace(trace, "metadata_enricher.rule", {"chunk_id": chunk.id})
            return _copy_chunk(chunk, metadata)

        llm_metadata = self._llm_enrich(chunk.text, trace)
        if llm_metadata is None:
            metadata["metadata_fallback_reason"] = "llm_enrichment_failed"
            _record_trace(trace, "metadata_enricher.fallback", {"chunk_id": chunk.id})
            return _copy_chunk(chunk, metadata)

        metadata.update(llm_metadata)
        metadata["metadata_enriched_by"] = "llm"
        _record_trace(trace, "metadata_enricher.llm", {"chunk_id": chunk.id})
        return _copy_chunk(chunk, metadata)

    def _llm_enrich(self, text: str, trace: Any | None = None) -> dict[str, Any] | None:
        """Return structured metadata from an LLM, or None on any failure."""
        try:
            llm = self.llm or LLMFactory.create(self.settings)
            response = llm.chat(
                [
                    ChatMessage(
                        role="system",
                        content=(
                            "Extract retrieval metadata as strict JSON with keys "
                            "title, summary, tags. tags must be a list of short strings."
                        ),
                    ),
                    ChatMessage(role="user", content=f"Chunk:\n{text}"),
                ]
            )
            decoded = json.loads(response)
        except Exception:
            return None
        return _validate_llm_metadata(decoded)

    def _use_llm(self) -> bool:
        raw = self.settings.raw or {}
        ingestion = raw.get("ingestion") if isinstance(raw, dict) else None
        metadata_enricher = ingestion.get("metadata_enricher") if isinstance(ingestion, dict) else None
        if isinstance(metadata_enricher, dict) and "use_llm" in metadata_enricher:
            return bool(metadata_enricher["use_llm"])
        return False


def _rule_metadata(text: str) -> dict[str, Any]:
    clean = _clean_text(text)
    title = _derive_title(clean)
    summary = _derive_summary(clean)
    tags = _derive_tags(clean)
    return {"title": title, "summary": summary, "tags": tags}


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[IMAGE:\s*[A-Za-z0-9_.:-]+\]", " ", text)
    text = re.sub(r"[#*_>`~\[\]()<>{}|]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _derive_title(text: str) -> str:
    if not text:
        return "Untitled Chunk"
    first_sentence = re.split(r"(?<=[.!?。！？])\s+", text, maxsplit=1)[0].strip()
    words = first_sentence.split()
    title = " ".join(words[:10]).strip(" .。")
    return title or "Untitled Chunk"


def _derive_summary(text: str) -> str:
    if not text:
        return "No summary available."
    words = text.split()
    summary = " ".join(words[:40]).strip()
    return summary or "No summary available."


def _derive_tags(text: str) -> list[str]:
    words = [
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)
        if word.lower() not in _STOP_WORDS
    ]
    common = [word for word, _ in Counter(words).most_common(5)]
    return common or ["general"]


def _validate_llm_metadata(decoded: Any) -> dict[str, Any] | None:
    if not isinstance(decoded, dict):
        return None
    title = decoded.get("title")
    summary = decoded.get("summary")
    tags = decoded.get("tags")
    if not isinstance(title, str) or not title.strip():
        return None
    if not isinstance(summary, str) or not summary.strip():
        return None
    if not isinstance(tags, list) or not tags or not all(isinstance(tag, str) and tag.strip() for tag in tags):
        return None
    return {"title": title.strip(), "summary": summary.strip(), "tags": [tag.strip() for tag in tags]}


def _copy_chunk(chunk: Chunk, metadata: dict[str, Any]) -> Chunk:
    return Chunk(
        id=chunk.id,
        text=chunk.text,
        metadata=metadata,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
        source_ref=chunk.source_ref,
    )


def _record_trace(trace: Any | None, name: str, data: dict[str, Any]) -> None:
    if hasattr(trace, "record_stage"):
        trace.record_stage(name, data)


_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "about",
    "chunk",
    "section",
    "document",
}
