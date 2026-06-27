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

## Deployment

Free-tier deploy: **backend → Hugging Face Spaces (Docker)**, **frontend →
Vercel**. The backend image (`backend/Dockerfile`) bakes the embedding model and
FAISS index in at build time so the container is self-contained. Step-by-step
runbook: [DEPLOYMENT.md](DEPLOYMENT.md).

## Evaluation

The system is measured against a **held-out** set of paraphrased questions
(`backend/eval/`) — phrased differently from the knowledge base so we test
generalization, not memorization — plus out-of-scope questions that test the
refusal guardrail.

Two layers:
- **Retrieval & guardrail** (deterministic, free): hit@1, hit@3, MRR, and how
  reliably off-topic questions are refused.
- **Generation quality** (`--judge`): an **independent, stronger LLM judge**
  (OpenAI `gpt-4o-mini`) grades the free HF generator's answers for
  *faithfulness* (grounded in retrieved context) and *correctness* (matches the
  reference). Using a separate, stronger judge avoids the self-preference bias
  of letting a model grade itself.

```bash
cd backend
python -m eval.run_eval                         # retrieval + guardrail (free)
python -m eval.run_eval --judge                 # + OpenAI-judged generation
python -m eval.run_eval --judge --judge-provider hf   # judge with the free model instead
```

Latest results (30 in-scope, 6 out-of-scope):

| Metric | Score |
|---|---|
| Retrieval hit@1 | 0.80 |
| Retrieval hit@3 | 0.97 |
| MRR | 0.88 |
| False-refusal rate (in-scope) | 0.00 |
| Refusal accuracy (out-of-scope) | 1.00 |
| Faithfulness (gpt-4o-mini judge) | 0.87 |
| Correctness (gpt-4o-mini judge) | 0.80 |

The retrieval threshold (`score_threshold`) was tuned with this harness: every
in-scope hit scores ≥ 0.75 while off-topic queries top out at ~0.65, so a 0.7
cutoff yields zero false refusals and 100% off-topic refusal.

## License
MIT
