"""Query preprocessing for retrieval keywords and metadata filters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


class QueryProcessorError(ValueError):
    """Raised when query preprocessing input is invalid."""


@dataclass(frozen=True)
class ProcessedQuery:
    """Normalized query contract consumed by retrievers."""

    original_query: str
    normalized_query: str
    keywords: list[str]
    filters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the processed query with stable field names."""
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "keywords": list(self.keywords),
            "filters": dict(self.filters),
        }


class QueryProcessor:
    """Extract retrieval keywords and simple metadata filters from a raw query."""

    def __init__(self, stop_words: set[str] | None = None, min_keyword_length: int = 2) -> None:
        if min_keyword_length <= 0:
            raise QueryProcessorError("min_keyword_length must be greater than 0")
        self.stop_words = set(_DEFAULT_STOP_WORDS if stop_words is None else stop_words)
        self.min_keyword_length = min_keyword_length

    def process(
        self,
        query: str,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> ProcessedQuery:
        """Return normalized keywords plus merged explicit and inline filters."""
        if not isinstance(query, str) or not query.strip():
            raise QueryProcessorError("query must be a non-empty string")
        explicit_filters = _normalize_filters(filters)
        inline_filters, query_without_filters = _extract_inline_filters(query)
        merged_filters = {**inline_filters, **explicit_filters}
        normalized_query = _normalize_query_text(query_without_filters)
        keywords = self.extract_keywords(normalized_query)
        if not keywords:
            keywords = self.extract_keywords(query)
        if not keywords:
            keywords = _non_ascii_keyword_fallback(normalized_query)
        if not keywords:
            raise QueryProcessorError("query produced no keywords")

        result = ProcessedQuery(
            original_query=query,
            normalized_query=normalized_query,
            keywords=keywords,
            filters=merged_filters,
        )
        _record_trace(trace, "query_processor.processed", {"keyword_count": len(keywords), "filters": merged_filters})
        return result

    def extract_keywords(self, text: str) -> list[str]:
        """Tokenize text into stable, deduplicated keyword terms."""
        if not isinstance(text, str):
            raise QueryProcessorError("text must be a string")
        tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", text)
            if len(token) >= self.min_keyword_length
        ]
        keywords: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token in self.stop_words or token in seen:
                continue
            seen.add(token)
            keywords.append(token)
        return keywords


def _extract_inline_filters(query: str) -> tuple[dict[str, Any], str]:
    filters: dict[str, Any] = {}

    def replace(match: re.Match[str]) -> str:
        key = match.group("key").lower()
        raw_value = match.group("quoted") or match.group("single") or match.group("bare") or ""
        filters[key] = _coerce_filter_value(raw_value)
        return " "

    stripped_query = _INLINE_FILTER_PATTERN.sub(replace, query)
    return filters, stripped_query


def _normalize_filters(filters: Mapping[str, Any] | None) -> dict[str, Any]:
    if filters is None:
        return {}
    if not isinstance(filters, Mapping):
        raise QueryProcessorError("filters must be a mapping")
    normalized: dict[str, Any] = {}
    for key, value in filters.items():
        if not isinstance(key, str) or not key.strip():
            raise QueryProcessorError("filter keys must be non-empty strings")
        if value is None or value == "":
            continue
        normalized[key.strip().lower()] = value.strip() if isinstance(value, str) else value
    return normalized


def _normalize_query_text(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def _non_ascii_keyword_fallback(query: str) -> list[str]:
    normalized = _normalize_query_text(query)
    if not normalized:
        return []
    if any(not char.isascii() and not char.isspace() for char in normalized):
        return [normalized]
    return []


def _coerce_filter_value(value: str) -> str | bool | int | float:
    value = value.strip().strip("'\"")
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def _record_trace(trace: Any | None, name: str, data: dict[str, Any]) -> None:
    if hasattr(trace, "record_stage"):
        trace.record_stage(name, data)


_INLINE_FILTER_PATTERN = re.compile(
    r"(?P<key>collection|doc_type|language|access_level|source_path|tag|author|year)"
    r"\s*(?:=|:)\s*(?:\"(?P<quoted>[^\"]+)\"|'(?P<single>[^']+)'|(?P<bare>[^\s]+))",
    flags=re.IGNORECASE,
)


_DEFAULT_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}
