"""Volcengine Ark OpenAI-compatible chat completion provider."""

from __future__ import annotations

from libs.llm.openai_llm import OpenAICompatibleLLM


class ArkLLM(OpenAICompatibleLLM):
    """Volcengine Ark chat completion client using the OpenAI-compatible API."""

    provider_name = "ark"
    default_base_url = "https://ark.cn-beijing.volces.com/api/v3"
