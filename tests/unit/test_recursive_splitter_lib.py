"""Tests for the default recursive splitter implementation."""

from __future__ import annotations

import pytest

from core.settings import SplitterSettings
from libs.splitter import RecursiveSplitter, SplitterError, SplitterFactory


@pytest.fixture(autouse=True)
def register_recursive_splitter() -> None:
    SplitterFactory.clear()
    SplitterFactory.register("recursive", RecursiveSplitter)


def test_factory_creates_recursive_splitter() -> None:
    splitter = SplitterFactory.create(
        SplitterSettings(strategy="recursive", chunk_size=120, chunk_overlap=0)
    )

    assert isinstance(splitter, RecursiveSplitter)


def test_split_text_returns_empty_for_blank_text() -> None:
    splitter = RecursiveSplitter(SplitterSettings(strategy="recursive", chunk_size=120, chunk_overlap=0))

    assert splitter.split_text("  \n\n ") == []


def test_markdown_headings_start_new_chunks_when_needed() -> None:
    splitter = RecursiveSplitter(SplitterSettings(strategy="recursive", chunk_size=60, chunk_overlap=0))
    text = "# Intro\nShort intro.\n\n## Details\nMore detail about the system."

    chunks = splitter.split_text(text)

    assert chunks == ["# Intro\nShort intro.", "## Details\nMore detail about the system."]


def test_fenced_code_block_is_not_broken() -> None:
    splitter = RecursiveSplitter(SplitterSettings(strategy="recursive", chunk_size=80, chunk_overlap=0))
    text = "Before\n\n```python\nprint('hello')\nprint('world')\n```\n\nAfter"

    chunks = splitter.split_text(text)

    code_chunks = [chunk for chunk in chunks if "```python" in chunk]
    assert len(code_chunks) == 1
    assert "print('hello')\nprint('world')" in code_chunks[0]
    assert code_chunks[0].count("```") == 2


def test_long_paragraph_is_split_under_chunk_size() -> None:
    splitter = RecursiveSplitter(SplitterSettings(strategy="recursive", chunk_size=35, chunk_overlap=0))
    text = "alpha beta gamma delta epsilon zeta eta theta"

    chunks = splitter.split_text(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 35 for chunk in chunks)
    assert " ".join(chunks).replace("  ", " ") == text


def test_overlap_adds_tail_context_to_next_chunk() -> None:
    splitter = RecursiveSplitter(SplitterSettings(strategy="recursive", chunk_size=30, chunk_overlap=5))
    text = "first paragraph is here\n\nsecond paragraph follows"

    chunks = splitter.split_text(text)

    assert len(chunks) == 2
    assert chunks[1].startswith("here\n\nsecond")


def test_invalid_chunk_size_has_readable_error() -> None:
    with pytest.raises(SplitterError, match="chunk_size"):
        RecursiveSplitter(SplitterSettings(strategy="recursive", chunk_size=0, chunk_overlap=0))


def test_invalid_overlap_has_readable_error() -> None:
    with pytest.raises(SplitterError, match="chunk_overlap"):
        RecursiveSplitter(SplitterSettings(strategy="recursive", chunk_size=10, chunk_overlap=10))
