# Center Desk RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers residence-hall
**Center Desk** procedure questions. It is built as a decoupled, production-style
app: a **FastAPI** Python service does the retrieval + generation, and a
**Next.js / TypeScript** frontend streams the answer to the user.

## Architecture

```
┌─────────────────────┐        POST /chat (SSE stream)      ┌──────────────────────┐
│  Next.js + TS UI     │ ───────────────────────────────▶   │  FastAPI backend       │
│  (frontend/)         │ ◀───────  token stream  ─────────   │  (backend/)            │
└─────────────────────┘                                     │   ├─ fastembed (local) │
                                                             │   ├─ FAISS (cosine)    │
                                                             │   └─ HF Inference LLM  │
                                                             └──────────────────────┘
```

- **Embeddings:** `BAAI/bge-small-en-v1.5` via **fastembed** — runs locally (ONNX,
  no PyTorch, no per-query API cost).
- **Vector store:** FAISS with cosine similarity. The *question* is embedded as
  the retrieval key; the full Q&A pair is stored as the document.
- **Guardrail:** the LLM answers **only** from retrieved context; off-topic
  questions (cosine score below the threshold) are refused rather than
  hallucinated.
- **LLM:** Hugging Face Inference API (free tier), streamed token-by-token.
- **Knowledge base:** `backend/data/Center_Desk_Manual.csv` (220 Q&A entries).

## Repository layout

```
backend/
  app/
    config.py      # typed settings (Pydantic)
    ingest.py      # CSV -> FAISS index (local embeddings)
    retriever.py   # load index + cosine search + threshold filter
    rag.py         # retrieve -> grounded prompt -> stream
    main.py        # FastAPI: GET /health, POST /chat (SSE)
  data/Center_Desk_Manual.csv
  scripts/add_dataset_entries.py
  requirements.txt
frontend/
  app/page.tsx     # streaming chat UI (client component)
```

## Setup

### Backend

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt

# Configure secrets
copy .env.example .env        # then add your HF_TOKEN

# Build the vector index (downloads the embedding model once)
python -m app.ingest

# Run the API
uvicorn app.main:app --reload                       # http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm install
# .env.local sets NEXT_PUBLIC_API_URL (defaults to http://localhost:8000)
npm run dev                                          # http://localhost:3000
```

## API

| Method | Path      | Description                                    |
|--------|-----------|------------------------------------------------|
| GET    | `/health` | Liveness + indexed-entry count                 |
| POST   | `/chat`   | `{ "message": "..." }` → Server-Sent Event stream of answer tokens |

## License
MIT
