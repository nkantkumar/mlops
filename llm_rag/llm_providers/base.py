"""Base interface for LLM providers."""

from abc import ABC, abstractmethod
from enum import Enum


class ProviderName(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"


class LLMProvider(ABC):
    """Abstract base for LLM providers (OpenAI, Anthropic, Hugging Face)."""

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
    """Factory: return provider by name (openai, anthropic, huggingface)."""
    name = name.lower().strip()
    if name == ProviderName.OPENAI:
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(**kwargs)
    if name == ProviderName.ANTHROPIC:
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(**kwargs)
    if name == ProviderName.HUGGINGFACE:
        from .huggingface_provider import HuggingFaceProvider
        return HuggingFaceProvider(**kwargs)
    raise ValueError(f"Unknown provider: {name}. Use openai, anthropic, or huggingface.")
