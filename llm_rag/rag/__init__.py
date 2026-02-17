"""RAG (Retrieval-Augmented Generation) with LangChain and LlamaIndex."""

# Lazy imports so importing LangChainRAG alone doesn't pull in LlamaIndex/ChromaDB

def __getattr__(name: str):
    if name == "LangChainRAG":
        from .langchain_rag import LangChainRAG
        return LangChainRAG
    if name == "LlamaIndexRAG":
        from .llamaindex_rag import LlamaIndexRAG
        return LlamaIndexRAG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LangChainRAG", "LlamaIndexRAG"]
