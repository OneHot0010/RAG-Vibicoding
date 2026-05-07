"""LLM-backed reranker implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.settings import RerankSettings
from libs.llm.base_llm import BaseLLM, ChatMessage
from libs.reranker.base_reranker import BaseReranker, RerankCandidate, RerankResult


class LLMRerankerError(ValueError):
    """Raised when LLM reranking configuration or output is invalid."""


class LLMReranker(BaseReranker):
    """Rerank candidates by asking an LLM for structured ranked ids."""

    default_prompt_path = Path("config/prompts/rerank.txt")

    def __init__(
        self,
        settings: RerankSettings,
        llm: BaseLLM,
        prompt_template: str | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.prompt_template = prompt_template or _read_prompt(prompt_path or self.default_prompt_path)

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Any | None = None,
    ) -> list[RerankResult]:
        if not isinstance(query, str) or not query.strip():
            raise LLMRerankerError("query must be a non-empty string")
        if not candidates:
            return []

        prompt = self._build_prompt(query, candidates)
        raw_response = self.llm.chat(
            [
                ChatMessage(role="system", content=self.prompt_template),
                ChatMessage(role="user", content=prompt),
            ]
        )
        ranked_ids = _parse_ranked_ids(raw_response)
        return _rank_candidates(candidates, ranked_ids)

    def _build_prompt(self, query: str, candidates: list[RerankCandidate]) -> str:
        payload = {
            "query": query,
            "candidates": [
                {
                    "id": candidate.id,
                    "text": candidate.text,
                    "score": candidate.score,
                    "metadata": candidate.metadata,
                }
                for candidate in candidates
            ],
            "required_output_schema": {"ranked_ids": ["candidate-id"]},
        }
        return json.dumps(payload, ensure_ascii=False)


def _read_prompt(path: str | Path) -> str:
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise LLMRerankerError(f"rerank prompt not found: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise LLMRerankerError(f"rerank prompt is empty: {prompt_path}")
    return prompt


def _parse_ranked_ids(raw_response: str) -> list[str]:
    try:
        decoded = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise LLMRerankerError("LLM reranker response must be valid JSON") from exc
    if not isinstance(decoded, dict):
        raise LLMRerankerError("LLM reranker response must be a JSON object")
    ranked_ids = decoded.get("ranked_ids")
    if not isinstance(ranked_ids, list) or not all(isinstance(item, str) for item in ranked_ids):
        raise LLMRerankerError("LLM reranker response must contain ranked_ids: list[str]")
    if len(ranked_ids) != len(set(ranked_ids)):
        raise LLMRerankerError("LLM reranker ranked_ids must not contain duplicates")
    return ranked_ids


def _rank_candidates(candidates: list[RerankCandidate], ranked_ids: list[str]) -> list[RerankResult]:
    by_id = {candidate.id: candidate for candidate in candidates}
    unknown_ids = [item_id for item_id in ranked_ids if item_id not in by_id]
    if unknown_ids:
        raise LLMRerankerError(f"LLM reranker returned unknown candidate ids: {unknown_ids}")

    ordered_ids = list(ranked_ids)
    ordered_ids.extend(candidate.id for candidate in candidates if candidate.id not in ranked_ids)

    total = len(ordered_ids)
    results: list[RerankResult] = []
    for index, candidate_id in enumerate(ordered_ids):
        candidate = by_id[candidate_id]
        results.append(
            RerankResult(
                id=candidate.id,
                text=candidate.text,
                score=float(total - index),
                rank=index + 1,
                metadata=candidate.metadata,
            )
        )
    return results
