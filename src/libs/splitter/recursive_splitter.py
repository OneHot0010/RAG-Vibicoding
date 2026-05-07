"""Markdown-aware recursive text splitter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.settings import SplitterSettings
from libs.splitter.base_splitter import BaseSplitter


class SplitterError(ValueError):
    """Raised when splitter configuration or input is invalid."""


@dataclass(frozen=True)
class TextBlock:
    text: str
    atomic: bool = False


class RecursiveSplitter(BaseSplitter):
    """Split text recursively while preserving Markdown fenced code blocks."""

    def __init__(self, settings: SplitterSettings) -> None:
        if settings.chunk_size <= 0:
            raise SplitterError("splitter.chunk_size must be greater than 0")
        if settings.chunk_overlap < 0:
            raise SplitterError("splitter.chunk_overlap must be non-negative")
        if settings.chunk_overlap >= settings.chunk_size:
            raise SplitterError("splitter.chunk_overlap must be smaller than splitter.chunk_size")
        self.settings = settings

    def split_text(self, text: str, trace: Any | None = None) -> list[str]:
        if not isinstance(text, str):
            raise SplitterError("text must be a string")
        if not text.strip():
            return []

        blocks = _markdown_blocks(text)
        pieces: list[TextBlock] = []
        for block in blocks:
            if block.atomic or len(block.text) <= self.settings.chunk_size:
                pieces.append(block)
            else:
                pieces.extend(
                    TextBlock(piece)
                    for piece in _split_long_text(block.text, self.settings.chunk_size)
                    if piece.strip()
                )
        return _merge_blocks(pieces, self.settings.chunk_size, self.settings.chunk_overlap)


def _markdown_blocks(text: str) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    current: list[str] = []
    in_fence = False
    fence_marker = ""

    for line in text.splitlines():
        stripped = line.lstrip()
        is_fence = stripped.startswith("```") or stripped.startswith("~~~")

        if is_fence:
            marker = stripped[:3]
            if not in_fence:
                _flush_normal_block(blocks, current)
                current = [line]
                in_fence = True
                fence_marker = marker
                continue
            if marker == fence_marker:
                current.append(line)
                blocks.append(TextBlock("\n".join(current).strip(), atomic=True))
                current = []
                in_fence = False
                fence_marker = ""
                continue

        if in_fence:
            current.append(line)
            continue

        if not line.strip():
            _flush_normal_block(blocks, current)
            current = []
            continue

        if stripped.startswith("#") and current:
            _flush_normal_block(blocks, current)
            current = []
        current.append(line)

    if current:
        blocks.append(TextBlock("\n".join(current).strip(), atomic=in_fence))
    return blocks


def _flush_normal_block(blocks: list[TextBlock], current: list[str]) -> None:
    if current and "\n".join(current).strip():
        blocks.append(TextBlock("\n".join(current).strip(), atomic=False))


def _split_long_text(text: str, chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    for separator in ("\n\n", "\n", ". ", " "):
        if separator in text:
            parts = [part for part in text.split(separator) if part]
            split_parts: list[str] = []
            for part in parts:
                candidate = part if separator == " " else part.strip()
                split_parts.extend(_split_long_text(candidate, chunk_size))
            return _merge_plain_parts(split_parts, chunk_size, separator)

    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]


def _merge_plain_parts(parts: list[str], chunk_size: int, separator: str) -> list[str]:
    chunks: list[str] = []
    current = ""
    joiner = separator if separator != ". " else ". "
    for part in parts:
        candidate = part if not current else f"{current}{joiner}{part}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current.strip())
        current = part
    if current:
        chunks.append(current.strip())
    return chunks


def _merge_blocks(blocks: list[TextBlock], chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""

    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        candidate = text if not current else f"{current}\n\n{text}"
        if len(candidate) <= chunk_size or not current:
            current = candidate
            continue

        chunks.append(current)
        current = _with_overlap(current, text, chunk_overlap)

    if current:
        chunks.append(current)
    return chunks


def _with_overlap(previous: str, next_text: str, chunk_overlap: int) -> str:
    if chunk_overlap <= 0:
        return next_text
    overlap = previous[-chunk_overlap:].strip()
    if not overlap:
        return next_text
    candidate = f"{overlap}\n\n{next_text}"
    return candidate
