"""DeepSeek OpenAI-compatible chat completion provider."""

from __future__ import annotations

from libs.llm.openai_llm import OpenAICompatibleLLM


class DeepSeekLLM(OpenAICompatibleLLM):
    """DeepSeek chat completion client using OpenAI-compatible schema."""

    provider_name = "deepseek"
    default_base_url = "https://api.deepseek.com/v1"
