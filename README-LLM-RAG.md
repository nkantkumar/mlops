# LLM & RAG Project

A small project demonstrating **LangChain**, **LlamaIndex**, multiple **LLM APIs** (OpenAI, Anthropic, Hugging Face), **RAG** (retrieval-augmented generation), and **prompt engineering**.

## Setup

```bash
# From repo root
pip install -r requirements-llm.txt
cp .env.example .env
# Edit .env and add at least OPENAI_API_KEY (and others as needed)
```

## Project layout

```
llm_rag/
├── config.py              # Settings from .env (API keys, models)
├── prompts/
│   └── templates.py       # RAG QA, summarization, chain-of-thought templates
├── llm_providers/         # Multi-LLM abstraction
│   ├── base.py            # LLMProvider interface + get_provider()
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   └── huggingface_provider.py
└── rag/
    ├── langchain_rag.py   # RAG with LangChain (Chroma + prompt + LLM)
    └── llamaindex_rag.py  # RAG with LlamaIndex (Chroma + query engine)
```

## Run the demo

```bash
# Simple LLM call (default: OpenAI)
python run_llm_demo.py --provider openai

# With prompt-engineering examples (RAG-style + chain-of-thought)
python run_llm_demo.py --provider openai --prompts

# With RAG (LangChain + LlamaIndex)
python run_llm_demo.py --provider openai --rag

# Use Anthropic or Hugging Face
python run_llm_demo.py --provider anthropic
python run_llm_demo.py --provider huggingface --prompts
```

## Features

- **LangChain & LlamaIndex**: RAG pipelines with Chroma; same logic can be run with either stack.
- **LLM APIs**: Single interface (`get_provider(name)`) for OpenAI, Anthropic, and Hugging Face; each provider exposes LangChain and LlamaIndex-compatible LLMs.
- **RAG**: Ingest documents, embed with OpenAI (or HF), store in Chroma, then query with a chosen LLM using a RAG prompt template.
- **Prompt engineering**: Templates for RAG QA, summarization, and chain-of-thought in `llm_rag/prompts/templates.py`; used in the demo with `--prompts`.

## API keys

- **OpenAI**: Needed for the default provider and for embeddings in RAG. Get keys at [platform.openai.com](https://platform.openai.com).
- **Anthropic**: Optional; for `--provider anthropic`. [console.anthropic.com](https://console.anthropic.com).
- **Hugging Face**: Optional; for `--provider huggingface` and for local/HF embeddings. [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
