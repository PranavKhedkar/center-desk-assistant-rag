# Production image for the Center Desk RAG API.
# Designed for Hugging Face Spaces (Docker SDK) but is a standard container that
# also runs on Render, Fly.io, Cloud Run, etc.

FROM python:3.11-slim

# Hugging Face Spaces requires the container to run as a non-root user (uid 1000)
# and only write inside that user's home.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    # Keep the embedding-model cache inside the image so no download happens at
    # runtime (the model is baked in during build).
    FASTEMBED_CACHE_PATH=/home/user/app/.fastembed_cache \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /home/user/app

# Install dependencies first for better layer caching.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the application (see .dockerignore for what is excluded).
COPY --chown=user . .

# Bake the FAISS index + embedding model into the image. This needs no secrets
# (the embedding model is public) and makes startup fast and self-contained.
RUN python -m app.ingest

EXPOSE 7860

# Shell form so ${PORT} is expanded (HF Spaces uses 7860; other hosts set $PORT).
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}
