"""RAG chain: retrieval → prompt → LLM → answer with source citations."""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from app.config import (
    DEFAULT_PRESET,
    DEFAULT_TEMPLATE,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    PARAMETER_PRESETS,
    PROMPT_TEMPLATES,
    RETRIEVER_K,
)
from app.retriever import get_retriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_context_for_llm(docs: list) -> str:
    """Clean context for the LLM prompt – no metadata headers that confuse small models."""
    return "\n\n".join(doc.page_content for doc in docs)


def _format_context_for_eval(docs: list) -> str:
    """Context string with source labels, used only for faithfulness evaluation."""
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "?")
        parts.append(f"[{source} | chunk: {chunk_id}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _extract_sources(docs: list) -> list[dict[str, str]]:
    """Return a deduplicated list of {file, chunk_id} dicts."""
    seen: set[tuple[str, str]] = set()
    sources: list[dict[str, str]] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "?")
        key = (source, chunk_id)
        if key not in seen:
            seen.add(key)
            sources.append({"file": source, "chunk_id": chunk_id})
    return sources


def get_llm(preset_name: str = DEFAULT_PRESET) -> ChatOllama:
    """Instantiate an Ollama LLM with the chosen parameter preset."""
    params = PARAMETER_PRESETS.get(preset_name, PARAMETER_PRESETS[DEFAULT_PRESET])
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=params.temperature,
        top_p=params.top_p,
    )


# ---------------------------------------------------------------------------
# Public query function
# ---------------------------------------------------------------------------

def query(
    question: str,
    template_name: str = DEFAULT_TEMPLATE,
    preset_name: str = DEFAULT_PRESET,
    chat_history: str = "",
    k: int = RETRIEVER_K,
) -> dict[str, Any]:
    """
    Execute a RAG query and return a structured result.

    Returns
    -------
    dict with keys:
        answer      – the model's response
        sources     – list of {file, chunk_id} dicts
        latency_ms  – end-to-end time in milliseconds
        tokens_used – total tokens if reported by Ollama, else None
        context     – raw context string (used by evaluation for faithfulness)
    """
    retriever = get_retriever(k=k)
    docs = retriever.invoke(question)
    context = _format_context_for_llm(docs)          # clean text for the LLM
    context_with_ids = _format_context_for_eval(docs) # labelled text for evaluation
    sources = _extract_sources(docs)

    template_str = PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES[DEFAULT_TEMPLATE])
    llm = get_llm(preset_name)

    # Build the prompt – the conversational template requires chat_history
    if "{chat_history}" in template_str:
        prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=template_str,
        )
        prompt_text = prompt.format(
            context=context,
            question=question,
            chat_history=chat_history or "No previous conversation.",
        )
    else:
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template_str,
        )
        prompt_text = prompt.format(context=context, question=question)

    t0 = time.perf_counter()
    response = llm.invoke(prompt_text)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    answer: str = response.content

    tokens_used: int | None = None
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        tokens_used = response.usage_metadata.get("total_tokens")

    logger.info(
        "Query completed | template=%s preset=%s latency=%.0f ms tokens=%s",
        template_name, preset_name, latency_ms, tokens_used,
    )

    return {
        "answer": answer,
        "sources": sources,
        "latency_ms": latency_ms,
        "tokens_used": tokens_used,
        "context": context_with_ids,  # labelled context kept for faithfulness evaluation
    }
