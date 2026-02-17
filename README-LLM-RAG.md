# LLM & RAG Project

A project demonstrating **LangChain**, **LlamaIndex**, **Google Gemini** (LLM + embeddings), **RAG** (retrieval-augmented generation), and **prompt engineering**.

## Setup

```bash
# From repo root
pip install -r requirements-llm.txt
cp .env.example .env
# Edit .env and add GEMINI_API_KEY
```

## Project layout

```
llm_rag/
├── config.py              # Settings from .env (API key, models)
├── prompts/
│   └── templates.py       # RAG QA, summarization, chain-of-thought templates
├── llm_providers/         # LLM abstraction
│   ├── base.py            # LLMProvider interface + get_provider()
│   └── gemini_provider.py
└── rag/
    ├── langchain_rag.py   # RAG with LangChain (Chroma + prompt + LLM)
    └── llamaindex_rag.py  # RAG with LlamaIndex (Chroma + query engine)
```

## Run the demo

```bash
# Simple LLM call
python run_llm_demo.py

# With prompt-engineering examples (RAG-style + chain-of-thought)
python run_llm_demo.py --prompts

# With RAG (LangChain + LlamaIndex)
python run_llm_demo.py --rag

# All: prompts + RAG
python run_llm_demo.py --prompts --rag
```

## Features

- **LangChain & LlamaIndex**: RAG pipelines with Chroma; same logic can be run with either stack.
- **Gemini**: Single provider for chat and RAG embeddings via `get_provider("gemini")`.
- **RAG**: Ingest documents, embed with Gemini, store in Chroma, then query with Gemini using a RAG prompt template.
- **Prompt engineering**: Templates for RAG QA, summarization, and chain-of-thought in `llm_rag/prompts/templates.py`; used in the demo with `--prompts`.

## API key

- **Google Gemini**: For the LLM and for RAG embeddings. Get an API key at [Google AI Studio](https://aistudio.google.com/app/apikey).
