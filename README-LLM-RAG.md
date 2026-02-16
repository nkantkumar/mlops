# LLM & RAG Project

A project demonstrating **LangChain**, **LlamaIndex**, **LLM APIs** (Anthropic, Google Gemini), **RAG** (retrieval-augmented generation), and **prompt engineering**.

## Setup

```bash
# From repo root
pip install -r requirements-llm.txt
cp .env.example .env
# Edit .env and add GEMINI_API_KEY and/or ANTHROPIC_API_KEY
```

## Project layout

```
llm_rag/
├── config.py              # Settings from .env (API keys, models)
├── prompts/
│   └── templates.py       # RAG QA, summarization, chain-of-thought templates
├── llm_providers/         # Multi-LLM abstraction
│   ├── base.py            # LLMProvider interface + get_provider()
│   ├── anthropic_provider.py
│   └── gemini_provider.py
└── rag/
    ├── langchain_rag.py   # RAG with LangChain (Chroma + prompt + LLM)
    └── llamaindex_rag.py  # RAG with LlamaIndex (Chroma + query engine)
```

## Run the demo

```bash
# Simple LLM call (default: Gemini)
python run_llm_demo.py --provider gemini

# With prompt-engineering examples (RAG-style + chain-of-thought)
python run_llm_demo.py --provider gemini --prompts

# With RAG (LangChain + LlamaIndex)
python run_llm_demo.py --provider gemini --rag

# Use Anthropic
python run_llm_demo.py --provider anthropic
```

## Features

- **LangChain & LlamaIndex**: RAG pipelines with Chroma; same logic can be run with either stack.
- **LLM APIs**: Single interface (`get_provider(name)`) for Anthropic and Gemini; each provider exposes LangChain and LlamaIndex-compatible LLMs.
- **RAG**: Ingest documents, embed with Gemini, store in Chroma, then query with a chosen LLM using a RAG prompt template.
- **Prompt engineering**: Templates for RAG QA, summarization, and chain-of-thought in `llm_rag/prompts/templates.py`; used in the demo with `--prompts`.

## API keys

- **Anthropic**: For `--provider anthropic`. [console.anthropic.com](https://console.anthropic.com).
- **Google Gemini**: For `--provider gemini` and for RAG embeddings. Get an API key at [Google AI Studio](https://aistudio.google.com/app/apikey).
