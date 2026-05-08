"""Tests for MCP response construction and citations."""

from __future__ import annotations

import pytest

from core.response import CitationGenerator, NO_RESULTS_TEXT, ResponseBuilder
from core.types import RetrievalResult


def result(chunk_id: str = "chunk-a", score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text="Azure configuration uses endpoints and deployments.",
        metadata={"source_path": "docs/azure.pdf", "page": 3, "collection": "docs"},
    )


def test_citation_generator_emits_structured_fields() -> None:
    citations = CitationGenerator().generate([result()])

    assert [citation.to_dict() for citation in citations] == [
        {
            "index": 1,
            "source": "docs/azure.pdf",
            "page": 3,
            "chunk_id": "chunk-a",
            "score": 0.9,
        }
    ]


def test_response_builder_returns_markdown_with_citation_markers() -> None:
    response = ResponseBuilder().build([result("chunk-a", 0.9), result("chunk-b", 0.7)], "Azure config")

    payload = response.to_dict()

    assert payload["content"][0]["type"] == "text"
    assert "## Knowledge Hub Results" in payload["content"][0]["text"]
    assert "### [1] chunk-a" in payload["content"][0]["text"]
    assert "### [2] chunk-b" in payload["content"][0]["text"]
    assert payload["structuredContent"]["query"] == "Azure config"
    assert payload["structuredContent"]["citations"][0] == {
        "index": 1,
        "source": "docs/azure.pdf",
        "page": 3,
        "chunk_id": "chunk-a",
        "score": 0.9,
    }
    assert payload["structuredContent"]["results"][0]["chunk_id"] == "chunk-a"


def test_response_builder_no_results_is_friendly_message() -> None:
    payload = ResponseBuilder().build([], "missing").to_dict()

    assert payload["content"] == [{"type": "text", "text": NO_RESULTS_TEXT}]
    assert payload["structuredContent"] == {"query": "missing", "citations": [], "results": []}


def test_response_builder_validates_query_and_results() -> None:
    with pytest.raises(ValueError, match="query"):
        ResponseBuilder().build([result()], "")
    with pytest.raises(ValueError, match="RetrievalResult"):
        ResponseBuilder().build(["bad"], "query")  # type: ignore[list-item]
