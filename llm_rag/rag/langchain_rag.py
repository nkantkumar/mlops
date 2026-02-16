"""RAG pipeline using LangChain: retriever + prompt + LLM."""

from pathlib import Path
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm_rag.config import settings
from llm_rag.prompts.templates import RAG_QA_TEMPLATE


class LangChainRAG:
    """RAG using LangChain: Chroma retriever + custom prompt + any LangChain LLM."""

    def __init__(
        self,
        persist_dir: str | None = None,
        embedding_model: str | None = None,
        top_k: int = 4,
    ):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.embedding_model = embedding_model or settings.OPENAI_EMBEDDING_MODEL
        self.top_k = top_k
        self._embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            api_key=settings.OPENAI_API_KEY or None,
        )
        self._vectorstore: Chroma | None = None
        self._retriever = None

    def load_or_create_vectorstore(self, documents: List[str] | None = None):
        """Load existing Chroma store or create from documents."""
        path = Path(self.persist_dir)
        path.mkdir(parents=True, exist_ok=True)

        if documents:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = splitter.create_documents(documents)
            self._vectorstore = Chroma.from_documents(
                splits,
                self._embeddings,
                persist_directory=self.persist_dir,
            )
            self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        else:
            self._vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self._embeddings,
            )
            self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        return self

    def add_documents(self, documents: List[str]):
        """Add more documents to the existing vector store."""
        if not self._vectorstore:
            return self.load_or_create_vectorstore(documents)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.create_documents(documents)
        self._vectorstore.add_documents(splits)
        return self

    def query(self, question: str, llm) -> str:
        """Run RAG: retrieve context, format prompt, call LLM."""
        if not self._retriever:
            raise RuntimeError("Call load_or_create_vectorstore first.")
        prompt = ChatPromptTemplate.from_template(RAG_QA_TEMPLATE)
        chain = (
            {"context": self._retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain.invoke(question)
