"""Build MCP-compatible responses from retrieval results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.response.citation_generator import CitationGenerator
from core.response.multimodal_assembler import MultimodalAssembler
from core.types import RetrievalResult


NO_RESULTS_TEXT = "未找到相关文档，请先运行 ingest.py 摄取数据。"


@dataclass(frozen=True)
class MCPResponse:
    """MCP tool response payload."""

    content: list[dict[str, Any]]
    structuredContent: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize in MCP tools/call result shape."""
        return {"content": self.content, "structuredContent": self.structuredContent}


class ResponseBuilder:
    """Construct readable Markdown plus structured citations."""

    def __init__(
        self,
        citation_generator: CitationGenerator | None = None,
        multimodal_assembler: MultimodalAssembler | None = None,
    ) -> None:
        self.citation_generator = citation_generator or CitationGenerator()
        self.multimodal_assembler = multimodal_assembler or MultimodalAssembler()

    def build(self, retrieval_results: list[RetrievalResult], query: str) -> MCPResponse:
        """Build an MCP response for a query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not retrieval_results:
            return MCPResponse(
                content=[{"type": "text", "text": NO_RESULTS_TEXT}],
                structuredContent={"query": query, "citations": [], "results": []},
            )

        citations = self.citation_generator.generate(retrieval_results)
        text = _markdown_results(retrieval_results, citations, query)
        multimodal = self.multimodal_assembler.assemble(retrieval_results)
        return MCPResponse(
            content=[{"type": "text", "text": text}, *multimodal.to_content()],
            structuredContent={
                "query": query,
                "citations": [citation.to_dict() for citation in citations],
                "results": [result.to_dict() for result in retrieval_results],
                "multimodal": multimodal.to_dict(),
            },
        )


def _markdown_results(results: list[RetrievalResult], citations: list[Any], query: str) -> str:
    lines = [f"## Knowledge Hub Results", "", f"Query: {query}", ""]
    for result, citation in zip(results, citations):
        lines.extend(
            [
                f"### [{citation.index}] {result.chunk_id}",
                f"Score: {result.score:.6f}",
                _summarize(result.text),
                f"Source: {citation.source}" + (f" page {citation.page}" if citation.page is not None else ""),
                "",
            ]
        )
    return "\n".join(lines).strip()


def _summarize(text: str, limit: int = 700) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return f"{clean[: limit - 3]}..."
