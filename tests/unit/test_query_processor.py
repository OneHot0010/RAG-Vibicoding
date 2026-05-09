"""Tests for retrieval query preprocessing."""

from __future__ import annotations

import pytest

from core.query_engine import ProcessedQuery, QueryProcessor, QueryProcessorError
from core.trace.trace_context import TraceContext


def test_process_extracts_keywords_and_empty_filters_by_default() -> None:
    result = QueryProcessor().process("What is hybrid retrieval for PDF documents?")

    assert isinstance(result, ProcessedQuery)
    assert result.original_query == "What is hybrid retrieval for PDF documents?"
    assert result.normalized_query == "What is hybrid retrieval for PDF documents?"
    assert result.keywords == ["hybrid", "retrieval", "pdf", "documents"]
    assert result.filters == {}


def test_inline_filters_are_parsed_and_removed_from_keywords() -> None:
    result = QueryProcessor().process('hybrid search collection:docs doc_type="pdf" year=2026')

    assert result.filters == {"collection": "docs", "doc_type": "pdf", "year": 2026}
    assert result.normalized_query == "hybrid search"
    assert result.keywords == ["hybrid", "search"]


def test_explicit_filters_override_inline_filters_and_are_normalized() -> None:
    result = QueryProcessor().process(
        "indexing collection:notes language:zh",
        filters={"Collection": "docs", "access_level": "internal"},
    )

    assert result.filters == {"collection": "docs", "language": "zh", "access_level": "internal"}
    assert result.keywords == ["indexing"]


def test_keyword_extraction_is_stable_deduped_and_configurable() -> None:
    processor = QueryProcessor(stop_words={"skip"}, min_keyword_length=3)

    assert processor.extract_keywords("Alpha alpha ab skip beta-search") == ["alpha", "beta-search"]


def test_bool_and_float_filter_values_are_coerced() -> None:
    result = QueryProcessor().process("images tag:true year:2026 access_level=0.75")

    assert result.filters == {"tag": True, "year": 2026, "access_level": 0.75}
    assert result.keywords == ["images"]


def test_result_serializes_stable_shape() -> None:
    result = QueryProcessor().process("retrieval collection:docs")

    assert result.to_dict() == {
        "original_query": "retrieval collection:docs",
        "normalized_query": "retrieval",
        "keywords": ["retrieval"],
        "filters": {"collection": "docs"},
    }


def test_non_ascii_query_falls_back_to_whole_query_keyword() -> None:
    result = QueryProcessor().process("测试查询")

    assert result.normalized_query == "测试查询"
    assert result.keywords == ["测试查询"]


def test_trace_records_keyword_count_and_filters() -> None:
    trace = TraceContext()

    QueryProcessor().process("retrieval collection:docs", trace=trace)

    assert trace.stages == [
        {
            "name": "query_processor.processed",
            "data": {"keyword_count": 1, "filters": {"collection": "docs"}},
        }
    ]


def test_invalid_inputs_have_readable_errors() -> None:
    with pytest.raises(QueryProcessorError, match="query"):
        QueryProcessor().process("")
    with pytest.raises(QueryProcessorError, match="filters"):
        QueryProcessor().process("query", filters=["bad"])  # type: ignore[arg-type]
    with pytest.raises(QueryProcessorError, match="filter keys"):
        QueryProcessor().process("query", filters={"": "bad"})
    with pytest.raises(QueryProcessorError, match="min_keyword_length"):
        QueryProcessor(min_keyword_length=0)


def test_query_with_only_stop_words_is_rejected() -> None:
    with pytest.raises(QueryProcessorError, match="no keywords"):
        QueryProcessor().process("the and of")
