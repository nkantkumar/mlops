"""Hugging Face LLM provider (LangChain + LlamaIndex)."""

from llm_rag.config import settings
from .base import LLMProvider


class HuggingFaceProvider(LLMProvider):
    """Hugging Face Inference API or local models via LangChain and LlamaIndex."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or settings.HUGGINGFACE_MODEL
        self._api_key = api_key or settings.HUGGINGFACE_API_KEY

    def invoke(self, prompt: str, **kwargs) -> str:
        llm = self.get_langchain_llm()
        out = llm.invoke(prompt, **kwargs)
        return out if isinstance(out, str) else getattr(out, "content", str(out))

    def get_langchain_llm(self):
        from langchain_community.llms import HuggingFaceHub
        # HuggingFaceHub uses Inference API; requires HUGGINGFACE_API_KEY for gated models
        return HuggingFaceHub(
            repo_id=self.model,
            huggingfacehub_api_token=self._api_key or None,
            model_kwargs={"temperature": 0.2, "max_length": 512},
        )

    def get_llamaindex_llm(self):
        from llama_index.llms.huggingface import HuggingFaceLLM
        return HuggingFaceLLM(
            model_name=self.model,
            tokenizer_name=self.model,
            token=self._api_key or None,
            temperature=0.2,
        )
