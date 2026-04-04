# RAG-DocAnalysis

A Q&A chatbot that answers questions **only from your documents** and cites the exact chunks it used.

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM & Embeddings | Ollama (`gemma3:1b`, `nomic-embed-text`) |
| RAG Framework | LangChain |
| Vector Store | ChromaDB |
| API | FastAPI |
| Chat UI | Gradio |
| Containerisation | Docker Compose |

---

## How it works

```
Your Documents (txt/md)
        │
        ▼
  [Ingestion] Split into chunks → embed → store in ChromaDB
        │
  User asks a question
        │
        ▼
  [Retrieval] Top-4 similar chunks fetched from ChromaDB
        │
        ▼
  [Generation] Chunks passed as context to Gemma 3 via Ollama
        │
        ▼
  Answer + source citations (filename + chunk ID)
```

The UI and API both expose template and preset controls:
- **3 prompt templates** — `concise`, `detailed`, `conversational`
- **2 LLM presets** — `precise` (temp 0.1) and `creative` (temp 0.7)

---

## Running locally

**Prerequisites:** Ollama must be running with the required models pulled.

```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text:latest
```

**Install dependencies:**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate       # macOS / Linux

pip install uv
uv pip install -e .
```

**Ingest your documents** (drop `.txt` or `.md` files into `docs/`):

```bash
python main.py ingest
```

**Start the app:**

```bash
python main.py all
# FastAPI  → http://localhost:8000/docs
# Gradio   → http://localhost:7860
```

Or start them separately:

```bash
python main.py api   # API only
python main.py ui    # UI only
```

**Run the evaluation suite** (15 Q&A pairs, measures accuracy / faithfulness / latency):

```bash
python main.py evaluate
```

---

## Running with Docker

```bash
docker compose build
docker compose run --rm api python main.py ingest   # embed docs once
docker compose up
# API  → http://localhost:8000/docs
# UI   → http://localhost:7860
```

> Ollama must be running on the **host machine**. The containers reach it via `host.docker.internal:11434`.
