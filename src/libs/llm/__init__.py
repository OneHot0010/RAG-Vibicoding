"""LLM abstractions, providers, and factory helpers."""

from libs.llm.azure_llm import AzureLLM
from libs.llm.base_llm import BaseLLM, ChatMessage
from libs.llm.deepseek_llm import DeepSeekLLM
from libs.llm.llm_factory import LLMFactory, LLMFactoryError
from libs.llm.ollama_llm import OllamaLLM
from libs.llm.openai_llm import LLMProviderError, OpenAICompatibleLLM, OpenAILLM

LLMFactory.register("openai", OpenAILLM)
LLMFactory.register("azure", AzureLLM)
LLMFactory.register("deepseek", DeepSeekLLM)
LLMFactory.register("ollama", OllamaLLM)

__all__ = [
    "AzureLLM",
    "BaseLLM",
    "ChatMessage",
    "DeepSeekLLM",
    "LLMFactory",
    "LLMFactoryError",
    "LLMProviderError",
    "OllamaLLM",
    "OpenAICompatibleLLM",
    "OpenAILLM",
]
