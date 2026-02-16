"""Prompt engineering templates for RAG, summarization, and reasoning."""

# RAG: answer from retrieved context only
RAG_QA_TEMPLATE = """You are a helpful assistant. Answer the question based ONLY on the following context. If the context does not contain enough information, say so. Do not make up facts.

Context:
{context}

Question: {question}

Answer:"""

# Summarization with constraints
SUMMARIZE_TEMPLATE = """Summarize the following text in {num_sentences} sentences. Keep the tone {tone}. Focus on: {focus_areas}.

Text:
{text}

Summary:"""

# Chain-of-thought style reasoning
CHAIN_OF_THOUGHT_TEMPLATE = """Think step by step, then give a final answer.

Question: {question}

Let's think step by step:"""


def format_rag_prompt(context: str, question: str, template: str | None = None) -> str:
    """Format context and question into the RAG QA template."""
    t = template or RAG_QA_TEMPLATE
    return t.format(context=context.strip(), question=question.strip())
