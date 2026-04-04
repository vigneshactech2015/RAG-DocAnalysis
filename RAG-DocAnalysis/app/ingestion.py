"""Document ingestion pipeline: load → chunk → embed → store in ChromaDB."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_PERSIST_DIR,
    DOCS_DIR,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk_id(source: str, index: int, content: str) -> str:
    """Stable, human-readable chunk ID derived from source file and position."""
    stem = Path(source).stem
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{stem}_{index:04d}_{content_hash}"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_documents(docs_dir: str = DOCS_DIR) -> List[Document]:
    """Load all .txt and .md files from *docs_dir*, setting 'source' metadata."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    all_docs: List[Document] = []
    for pattern in ("**/*.txt", "**/*.md"):
        for filepath in sorted(docs_path.glob(pattern)):
            loader = TextLoader(str(filepath), encoding="utf-8")
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = filepath.name
            all_docs.extend(pages)
            logger.info("Loaded '%s' (%d page(s))", filepath.name, len(pages))

    logger.info("Total documents loaded: %d", len(all_docs))
    return all_docs


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks and assign stable chunk IDs."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        chunk.metadata["chunk_id"] = _make_chunk_id(source, i, chunk.page_content)
    logger.info("Total chunks created: %d", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Embed + Store
# ---------------------------------------------------------------------------

def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


def build_vectorstore(chunks: List[Document]) -> Chroma:
    """Embed chunks and persist them to a new ChromaDB collection."""
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )
    logger.info("Vector store built and persisted at '%s'", CHROMA_PERSIST_DIR)
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load an existing persisted ChromaDB collection."""
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings(),
    )


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def ingest(docs_dir: str = DOCS_DIR) -> Chroma:
    """Full ingestion pipeline: load → split → embed → persist."""
    docs = load_documents(docs_dir)
    chunks = split_documents(docs)
    return build_vectorstore(chunks)
