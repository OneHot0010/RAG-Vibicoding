"""Base abstractions for chat-oriented LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ChatMessage:
    """A normalized chat message passed to LLM providers."""

    role: str
    content: str


class BaseLLM(ABC):
    """Common interface for all LLM backends."""

    @abstractmethod
    def chat(self, messages: Sequence[ChatMessage | Mapping[str, Any]]) -> str:
        """Return a text response for a list of chat messages."""
