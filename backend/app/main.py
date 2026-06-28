"""FastAPI entrypoint exposing the RAG service.

Endpoints:
  GET  /health  -> liveness + index status (for deploy health checks)
  POST /chat     -> streams the assistant answer as Server-Sent Events (SSE)

The frontend (Next.js) calls /chat and renders tokens as they arrive. We make a
single non-streaming LLM call and then stream the finished text to the browser
word-by-word. Upstream token-streaming from the free HF Inference API proved
unreliable (intermittent httpx.StreamClosed), so this keeps the typing-effect UX
while removing the flaky dependency — the LLM call is the same one validated at
30/30 in the eval harness.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.rag import answer_text
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
            # answer_text is blocking (network I/O) — run it off the event loop.
            answer = await run_in_threadpool(answer_text, question)
            if not answer:
                answer = "I don't have that procedure in my knowledge base."
            # Stream word-by-word so the UI still "types" the answer out. Words
            # carry no whitespace/newlines, so there are no SSE framing issues;
            # the frontend re-joins them with spaces.
            for word in answer.split():
                yield {"data": word}
        except Exception as e:
            yield {"data": f"[error] {e}"}

    return EventSourceResponse(event_generator())
