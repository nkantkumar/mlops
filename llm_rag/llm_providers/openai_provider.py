"""OpenAI LLM provider (LangChain + LlamaIndex)."""

from llm_rag.config import settings
from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI API via LangChain and LlamaIndex."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or settings.OPENAI_MODEL
        self._api_key = api_key or settings.OPENAI_API_KEY

    def invoke(self, prompt: str, **kwargs) -> str:
        llm = self.get_langchain_llm()
        return llm.invoke(prompt, **kwargs).content

    def get_langchain_llm(self, temperature: float = 0.2):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=self.model,
            api_key=self._api_key or None,  # None lets it use OPENAI_API_KEY from env
            temperature=temperature,
        )

    def get_llamaindex_llm(self):
        from llama_index.llms.openai import OpenAI
        return OpenAI(
            model=self.model,
            api_key=self._api_key or None,
            temperature=0.2,
        )
