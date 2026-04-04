"""Retriever factory backed by the persisted ChromaDB vector store."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.vectorstores import VectorStoreRetriever

from app.config import CHROMA_PERSIST_DIR, RETRIEVER_K
from app.ingestion import load_vectorstore

logger = logging.getLogger(__name__)

# Module-level cache so we don't reload ChromaDB on every request.
_vectorstore = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        if not Path(CHROMA_PERSIST_DIR).exists():
            raise RuntimeError(
                f"Vector store not found at '{CHROMA_PERSIST_DIR}'. "
                "Run  python main.py ingest  first."
            )
        _vectorstore = load_vectorstore()
        logger.info("Vector store loaded from '%s'", CHROMA_PERSIST_DIR)
    return _vectorstore


def get_retriever(k: int = RETRIEVER_K) -> VectorStoreRetriever:
    """Return a similarity-search retriever over the persisted vector store."""
    return get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
