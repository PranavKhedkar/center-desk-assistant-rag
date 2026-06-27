---
title: Center Desk RAG API
emoji: 🛎️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Center Desk RAG API

FastAPI backend for the Center Desk RAG assistant. Built from the `Dockerfile`
in this repo. See the main project repository for the full architecture.

Endpoints:
- `GET /health` — liveness + indexed-entry count
- `POST /chat` — `{ "message": "..." }` → Server-Sent Event stream of answer tokens

## Required Space secrets (Settings → Variables and secrets)
- `HF_TOKEN` (secret) — Hugging Face Inference API token for the LLM.
- `CORS_ORIGINS` (variable) — your deployed frontend origin, e.g.
  `https://your-app.vercel.app` (comma-separate multiple).
