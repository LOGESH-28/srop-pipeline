# ── Stage: runtime ────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Install curl (needed for the docker-compose healthcheck) and clean up in one
# layer to keep the image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencies (cached layer — only re-runs when requirements.txt changes) ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY app/ ./app/

# ── Persistent data directory ─────────────────────────────────────────────────
# Mounted as a volume at runtime (see docker-compose.yml).
# Contains:
#   /data/srop.db           — SQLite database
#   /data/vector_store/     — ChromaDB persistent store
RUN mkdir -p /data

EXPOSE 8000

# Run as a non-root user for security
RUN adduser --disabled-password --gecos "" srop \
    && chown -R srop:srop /app /data
USER srop

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
