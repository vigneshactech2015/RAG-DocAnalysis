"""FastAPI server exposing the RAG pipeline as a REST API.

Endpoints
---------
GET  /health          – liveness check
POST /ingest          – (re-)ingest documents into the vector store
POST /ask             – run a RAG query; returns answer, sources, latency, tokens
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import (
    DEFAULT_PRESET,
    DEFAULT_TEMPLATE,
    PARAMETER_PRESETS,
    PROMPT_TEMPLATES,
)
from app.rag_chain import query as rag_query

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: pre-load the vector store once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from app.retriever import get_vectorstore
        get_vectorstore()
        logger.info("Vector store pre-loaded successfully.")
    except RuntimeError as exc:
        logger.warning("Vector store not available at startup: %s", exc)
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG-DocAnalysis API",
    description="Q&A over your document set, powered by LangChain + Ollama.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer")
    template: str = Field(
        DEFAULT_TEMPLATE,
        description=f"Prompt template: {list(PROMPT_TEMPLATES.keys())}",
    )
    preset: str = Field(
        DEFAULT_PRESET,
        description=f"LLM parameter preset: {list(PARAMETER_PRESETS.keys())}",
    )
    chat_history: str = Field(
        "",
        description="Formatted prior conversation (used by the 'conversational' template)",
    )
    k: int = Field(4, ge=1, le=20, description="Number of chunks to retrieve")


class SourceCitation(BaseModel):
    file: str
    chunk_id: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    latency_ms: float
    tokens_used: int | None


class IngestResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str
    vector_store_ready: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    from pathlib import Path
    from app.config import CHROMA_PERSIST_DIR
    vs_ready = Path(CHROMA_PERSIST_DIR).exists()
    return HealthResponse(status="ok", vector_store_ready=vs_ready)


@app.post("/ingest", response_model=IngestResponse, tags=["ops"])
async def ingest_documents(background_tasks: BackgroundTasks):
    """Trigger document ingestion in the background."""
    def _run_ingest():
        from app.ingestion import ingest
        ingest()

    background_tasks.add_task(_run_ingest)
    return IngestResponse(message="Ingestion started in the background.")


@app.post("/ask", response_model=AskResponse, tags=["qa"])
async def ask(request: AskRequest):
    """Answer a question from the ingested document set, with source citations."""
    if request.template not in PROMPT_TEMPLATES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown template '{request.template}'. "
                   f"Choose from: {list(PROMPT_TEMPLATES.keys())}",
        )
    if request.preset not in PARAMETER_PRESETS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown preset '{request.preset}'. "
                   f"Choose from: {list(PARAMETER_PRESETS.keys())}",
        )

    try:
        result = rag_query(
            question=request.question,
            template_name=request.template,
            preset_name=request.preset,
            chat_history=request.chat_history,
            k=request.k,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return AskResponse(
        answer=result["answer"],
        sources=[SourceCitation(**s) for s in result["sources"]],
        latency_ms=result["latency_ms"],
        tokens_used=result["tokens_used"],
    )
