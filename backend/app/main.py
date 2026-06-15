"""FastAPI entrypoint exposing the RAG service.

Endpoints:
  GET  /health  -> liveness + index status (for deploy health checks)
  POST /chat     -> streams the assistant answer as Server-Sent Events (SSE)

The frontend (Next.js) calls /chat and renders tokens as they arrive. SSE is
used instead of plain JSON so the answer streams token-by-token, matching the
chat UX of production assistants.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.config import settings
from app.rag import answer_stream
from app.retriever import get_retriever

app = FastAPI(title="Center Desk RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
def health() -> dict:
    try:
        n = len(get_retriever()._metadata)
        return {"status": "ok", "indexed_entries": n, "model": settings.hf_model}
    except Exception as e:  # index missing / not built yet
        return {"status": "degraded", "error": str(e)}


@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.message.strip()
    if len(question.split()) < 3:
        async def short():
            yield {"data": "Please ask a more detailed question about Center Desk procedures."}
        return EventSourceResponse(short())

    async def event_generator():
        try:
            for token in answer_stream(question):
                yield {"data": token}
        except Exception as e:
            yield {"data": f"[error] {e}"}

    return EventSourceResponse(event_generator())
