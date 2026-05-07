"""Chunk refinement transform with rule and optional LLM modes."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from core.settings import Settings
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform, TransformError
from libs.llm import BaseLLM, ChatMessage, LLMFactory


DEFAULT_CHUNK_REFINEMENT_PROMPT = Path("config/prompts/chunk_refinement.txt")


class ChunkRefiner(BaseTransform):
    """Clean noisy chunks and optionally ask an LLM for semantic refinement."""

    def __init__(
        self,
        settings: Settings,
        llm: BaseLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.prompt_template = self._load_prompt(prompt_path)

    def transform(self, chunks: list[Chunk], trace: Any | None = None) -> list[Chunk]:
        """Refine chunks independently so one bad chunk does not stop ingestion."""
        refined_chunks: list[Chunk] = []
        for chunk in chunks:
            try:
                refined_chunks.append(self._transform_one(chunk, trace))
            except Exception as exc:
                metadata = copy.deepcopy(chunk.metadata)
                metadata["refined_by"] = "original"
                metadata["refine_error"] = str(exc)
                refined_chunks.append(_copy_chunk(chunk, chunk.text, metadata))
        return refined_chunks

    def _transform_one(self, chunk: Chunk, trace: Any | None) -> Chunk:
        rule_text = self._rule_based_refine(chunk.text)
        metadata = copy.deepcopy(chunk.metadata)
        metadata["refined_by"] = "rule"

        if not self._use_llm():
            _record_trace(trace, "chunk_refiner.rule", {"chunk_id": chunk.id})
            return _copy_chunk(chunk, rule_text, metadata)

        llm_text = self._llm_refine(rule_text, trace)
        if llm_text:
            metadata["refined_by"] = "llm"
            _record_trace(trace, "chunk_refiner.llm", {"chunk_id": chunk.id})
            return _copy_chunk(chunk, llm_text, metadata)

        metadata["fallback_reason"] = "llm_refine_failed"
        _record_trace(trace, "chunk_refiner.fallback", {"chunk_id": chunk.id})
        return _copy_chunk(chunk, rule_text, metadata)

    def _rule_based_refine(self, text: str) -> str:
        """Remove common chunk noise while preserving fenced code blocks."""
        if not isinstance(text, str):
            raise TransformError("chunk text must be a string")

        segments = _split_fenced_code(text)
        refined: list[str] = []
        for segment, is_code in segments:
            refined.append(segment.rstrip() if is_code else _clean_plain_text(segment))
        return "\n\n".join(segment for segment in refined if segment.strip()).strip()

    def _llm_refine(self, text: str, trace: Any | None = None) -> str | None:
        """Return LLM-refined text, or None when LLM refinement fails."""
        try:
            llm = self.llm or LLMFactory.create(self.settings)
            prompt = self.prompt_template.format(text=text)
            response = llm.chat(
                [
                    ChatMessage(role="system", content="You refine retrieval chunks without adding facts."),
                    ChatMessage(role="user", content=prompt),
                ]
            )
        except Exception:
            return None
        if not isinstance(response, str) or not response.strip():
            return None
        return response.strip()

    def _load_prompt(self, prompt_path: str | Path | None = None) -> str:
        """Load the prompt template, falling back to a safe built-in template."""
        path = Path(prompt_path or DEFAULT_CHUNK_REFINEMENT_PROMPT)
        if not path.exists():
            return _default_prompt()
        prompt = path.read_text(encoding="utf-8").strip()
        if not prompt:
            return _default_prompt()
        if "{text}" not in prompt:
            prompt = f"{prompt}\n\nChunk:\n{{text}}"
        return prompt

    def _use_llm(self) -> bool:
        raw = self.settings.raw or {}
        ingestion = raw.get("ingestion") if isinstance(raw, dict) else None
        chunk_refiner = ingestion.get("chunk_refiner") if isinstance(ingestion, dict) else None
        if isinstance(chunk_refiner, dict) and "use_llm" in chunk_refiner:
            return bool(chunk_refiner["use_llm"])
        return False


def _copy_chunk(chunk: Chunk, text: str, metadata: dict[str, Any]) -> Chunk:
    return Chunk(
        id=chunk.id,
        text=text,
        metadata=metadata,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
        source_ref=chunk.source_ref,
    )


def _clean_plain_text(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"(?im)^\s*(page\s+\d+\s*(of\s+\d+)?|\d+\s*/\s*\d+)\s*$", "", text)
    text = re.sub(r"(?im)^\s*(confidential|draft|footer|header)\s*$", "", text)
    text = re.sub(r"(?m)^\s*[-=_*]{3,}\s*$", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _split_fenced_code(text: str) -> list[tuple[str, bool]]:
    pattern = re.compile(r"(```.*?```|~~~.*?~~~)", flags=re.DOTALL)
    segments: list[tuple[str, bool]] = []
    last = 0
    for match in pattern.finditer(text):
        if match.start() > last:
            segments.append((text[last : match.start()], False))
        segments.append((match.group(0), True))
        last = match.end()
    if last < len(text):
        segments.append((text[last:], False))
    return segments or [(text, False)]


def _record_trace(trace: Any | None, name: str, data: dict[str, Any]) -> None:
    if hasattr(trace, "record_stage"):
        trace.record_stage(name, data)


def _default_prompt() -> str:
    return (
        "Refine the following chunk for retrieval. Preserve factual meaning, remove noise, "
        "and return only the refined chunk text.\n\n{text}"
    )
