"""Central configuration for RAG-DocAnalysis.

All tuneable settings live here so changes never require editing business logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DOCS_DIR: str = os.getenv("DOCS_DIR", str(BASE_DIR / "docs"))
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma"))
EVAL_RESULTS_DIR: str = os.getenv("EVAL_RESULTS_DIR", str(BASE_DIR / "data" / "eval_results"))

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemma3:1b")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "4"))

# ---------------------------------------------------------------------------
# LLM parameter presets  (at least 2 required)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LLMParams:
    temperature: float
    top_p: float
    description: str = ""


PARAMETER_PRESETS: dict[str, LLMParams] = {
    "precise": LLMParams(
        temperature=0.1,
        top_p=0.9,
        description="Low temperature for factual, deterministic answers",
    ),
    "creative": LLMParams(
        temperature=0.7,
        top_p=0.95,
        description="Higher temperature for varied, expressive answers",
    ),
}

# ---------------------------------------------------------------------------
# Prompt templates  (at least 3 required)
# ---------------------------------------------------------------------------
PROMPT_TEMPLATES: dict[str, str] = {
    # Template 1 – concise, grounded answers
    "concise": (
        "Read the context below, then answer the question using only information from that context.\n"
        "Give a short, direct answer. "
        "If the context truly does not mention the topic, say 'Not covered in the provided documents.'\n\n"
        "### Context\n"
        "{context}\n\n"
        "### Question\n"
        "{question}\n\n"
        "### Answer\n"
    ),
    # Template 2 – detailed explanations with reasoning
    "detailed": (
        "You are a helpful assistant. Use the context passages below to write a thorough answer.\n"
        "Explain key points clearly and include relevant details from the context.\n"
        "Do not add information that is not in the context. "
        "If the topic is not in the context, say so clearly.\n\n"
        "### Context\n"
        "{context}\n\n"
        "### Question\n"
        "{question}\n\n"
        "### Detailed Answer\n"
    ),
    # Template 3 – conversational with memory of prior exchanges
    "conversational": (
        "You are a friendly, helpful assistant. Answer the user's question using "
        "only the context provided below. Keep prior conversation in mind for follow-up questions.\n"
        "If the answer is not in the context, say so politely.\n\n"
        "### Context\n"
        "{context}\n\n"
        "### Conversation so far\n"
        "{chat_history}\n\n"
        "### User\n"
        "{question}\n\n"
        "### Assistant\n"
    ),
}

DEFAULT_TEMPLATE: str = "concise"
DEFAULT_PRESET: str = "precise"
