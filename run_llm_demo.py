#!/usr/bin/env python3
"""
Demo: LangChain & LlamaIndex, multi-LLM (OpenAI, Anthropic, Hugging Face), RAG, prompt engineering.

Usage:
  pip install -r requirements-llm.txt
  cp .env.example .env   # add your API keys
  python run_llm_demo.py [--provider openai|anthropic|huggingface|gemini] [--rag] [--prompts]
"""

import argparse
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def demo_llm_providers(provider_name: str):
    """Call one LLM provider (OpenAI, Anthropic, Hugging Face, or Gemini)."""
    from llm_rag.llm_providers import get_provider

    provider = get_provider(provider_name)
    prompt = "What is 2+2? Answer in one short sentence."
    print(f"\n--- {provider_name.upper()} ---\nPrompt: {prompt}\n")
    try:
        response = provider.invoke(prompt)
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error: {e}\n(Missing API key? Check .env)\n")


def demo_prompt_engineering(provider_name: str):
    """Use prompt templates: RAG-style QA and chain-of-thought."""
    from llm_rag.llm_providers import get_provider
    from llm_rag.prompts.templates import (
        RAG_QA_TEMPLATE,
        CHAIN_OF_THOUGHT_TEMPLATE,
        format_rag_prompt,
    )

    provider = get_provider(provider_name)
    # RAG-style: context + question
    context = "Our company was founded in 2020. We sell cloud software."
    question = "When was the company founded?"
    rag_prompt = format_rag_prompt(context, question)
    print("\n--- Prompt engineering: RAG-style ---\n" + rag_prompt[:200] + "...\n")
    try:
        out = provider.invoke(rag_prompt)
        print("Response:", out, "\n")
    except Exception as e:
        print("Error:", e, "\n")

    # Chain-of-thought
    cot_prompt = CHAIN_OF_THOUGHT_TEMPLATE.format(question="What is 15 * 7?")
    print("--- Chain-of-thought prompt ---\n" + cot_prompt[:150] + "...\n")
    try:
        out = provider.invoke(cot_prompt)
        print("Response:", out, "\n")
    except Exception as e:
        print("Error:", e, "\n")


def demo_rag_langchain(provider_name: str):
    """RAG with LangChain: index a few docs, then ask a question."""
    from llm_rag.llm_providers import get_provider
    from llm_rag.rag import LangChainRAG

    docs = [
        "LangChain is a framework for building applications with LLMs. It supports many providers.",
        "LlamaIndex is a data framework for LLM applications. It focuses on indexing and retrieval.",
        "RAG stands for Retrieval-Augmented Generation. You retrieve relevant text then generate an answer.",
    ]
    rag = LangChainRAG(persist_dir="./data/chroma_lc", top_k=2)
    rag.load_or_create_vectorstore(documents=docs)

    provider = get_provider(provider_name)
    llm = provider.get_langchain_llm()
    question = "What is RAG?"
    print("\n--- LangChain RAG ---\nQuestion:", question, "\n")
    try:
        answer = rag.query(question, llm=llm)
        print("Answer:", answer, "\n")
    except Exception as e:
        print("Error:", e, "\n")


def demo_rag_llamaindex(provider_name: str):
    """RAG with LlamaIndex: set global LLM/embed model, build index, query."""
    from llama_index.core import Settings
    from llm_rag.llm_providers import get_provider
    from llm_rag.rag import LlamaIndexRAG
    from llm_rag.config import settings as app_settings

    # Set global LLM and embed model (required by LlamaIndex)
    provider = get_provider(provider_name)
    Settings.llm = provider.get_llamaindex_llm()
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding
        Settings.embed_model = OpenAIEmbedding(
            model=app_settings.OPENAI_EMBEDDING_MODEL,
            api_key=app_settings.OPENAI_API_KEY or None,
        )
    except Exception:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        Settings.embed_model = HuggingFaceEmbedding(model_name=app_settings.HF_EMBEDDING_MODEL)

    docs = [
        "Python 3.12 was released in October 2023. It has better error messages.",
        "Type hints in Python help IDEs and tools. Use typing module or built-in list[str].",
    ]
    rag = LlamaIndexRAG(persist_dir="./data/chroma_li", top_k=2)
    rag.load_or_create_index(documents=docs)

    question = "When was Python 3.12 released?"
    print("\n--- LlamaIndex RAG ---\nQuestion:", question, "\n")
    try:
        answer = rag.query(question)
        print("Answer:", answer, "\n")
    except Exception as e:
        print("Error:", e, "\n")


def main():
    parser = argparse.ArgumentParser(description="LLM & RAG demo")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "huggingface", "gemini"],
                        help="LLM provider to use")
    parser.add_argument("--rag", action="store_true", help="Run LangChain and LlamaIndex RAG demos")
    parser.add_argument("--prompts", action="store_true", help="Run prompt engineering demos")
    args = parser.parse_args()

    print("Provider:", args.provider)

    demo_llm_providers(args.provider)
    if args.prompts:
        demo_prompt_engineering(args.provider)
    if args.rag:
        demo_rag_langchain(args.provider)
        demo_rag_llamaindex(args.provider)

    print("Done.")


if __name__ == "__main__":
    main()
