"""Anthropic (Claude) LLM provider (LangChain + LlamaIndex)."""

from llm_rag.config import settings
from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Anthropic API via LangChain and LlamaIndex."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or settings.ANTHROPIC_MODEL
        self._api_key = api_key or settings.ANTHROPIC_API_KEY

    def invoke(self, prompt: str, **kwargs) -> str:
        llm = self.get_langchain_llm()
        return llm.invoke(prompt, **kwargs).content

    def get_langchain_llm(self):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=self.model,
            api_key=self._api_key or None,
            temperature=0.2,
        )

    def get_llamaindex_llm(self):
        from llama_index.llms.anthropic import Anthropic
        return Anthropic(
            model=self.model,
            api_key=self._api_key or None,
            temperature=0.2,
        )
