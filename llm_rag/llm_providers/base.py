"""Base interface for LLM providers."""

from abc import ABC, abstractmethod
from enum import Enum


class ProviderName(str, Enum):
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class LLMProvider(ABC):
    """Abstract base for LLM providers (Anthropic, Gemini)."""

    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> str:
        """Send prompt to the LLM and return the response text."""
        pass

    @abstractmethod
    def get_langchain_llm(self):
        """Return a LangChain-compatible LLM instance."""
        pass

    @abstractmethod
    def get_llamaindex_llm(self):
        """Return a LlamaIndex-compatible LLM instance."""
        pass


def get_provider(name: str, **kwargs) -> LLMProvider:
    """Factory: return provider by name (anthropic, gemini)."""
    name = name.lower().strip()
    if name == ProviderName.ANTHROPIC:
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(**kwargs)
    if name == ProviderName.GEMINI:
        from .gemini_provider import GeminiProvider
        return GeminiProvider(**kwargs)
    raise ValueError(f"Unknown provider: {name}. Use anthropic or gemini.")
