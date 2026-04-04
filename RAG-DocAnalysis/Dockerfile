# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# System deps: curl is used by the docker-compose health-check
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution, then install project deps
RUN pip install --no-cache-dir uv

COPY pyproject.toml .
RUN uv pip install --system --no-cache-dir .

# Copy application source
COPY . .

# Create persistent data directories with appropriate permissions
RUN mkdir -p data/chroma data/eval_results

# Expose ports: 8000 = FastAPI, 7860 = Gradio
EXPOSE 8000 7860

# Default: start both services (overridden in docker-compose per-service)
CMD ["python", "main.py", "all"]
