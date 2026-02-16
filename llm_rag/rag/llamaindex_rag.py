"""RAG pipeline using LlamaIndex: index + query engine."""

from pathlib import Path
from typing import List

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from llm_rag.config import settings


class LlamaIndexRAG:
    """RAG using LlamaIndex: Chroma vector store + global Settings LLM/embeddings."""

    def __init__(
        self,
        persist_dir: str | None = None,
        top_k: int = 4,
    ):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.top_k = top_k
        self._index: VectorStoreIndex | None = None
        self._chroma_client = None

    def _get_vector_store(self):
        path = Path(self.persist_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        chroma_collection = self._chroma_client.get_or_create_collection("llamaindex", metadata={"hnsw:space": "cosine"})
        return ChromaVectorStore(chroma_collection=chroma_collection)

    def load_or_create_index(self, documents: List[str] | None = None):
        """Create index from documents or load existing. Set Settings.llm and Settings.embed_model before use."""
        vector_store = self._get_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if documents:
            from llama_index.core import Document
            nodes = SentenceSplitter(chunk_size=500, chunk_overlap=50).get_nodes_from_documents(
                [Document(text=t) for t in documents]
            )
            self._index = VectorStoreIndex(nodes, storage_context=storage_context)
        else:
            self._index = VectorStoreIndex.from_vector_store(vector_store)
        return self

    def query(self, question: str) -> str:
        """Query the index; uses Settings.llm and Settings.embed_model."""
        if not self._index:
            raise RuntimeError("Call load_or_create_index first. Set Settings.llm and Settings.embed_model.")
        engine = self._index.as_query_engine(similarity_top_k=self.top_k)
        response = engine.query(question)
        return str(response)
