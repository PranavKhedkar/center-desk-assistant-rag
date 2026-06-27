"""RAG orchestration: retrieve context, build the prompt, stream the answer.

The grounding guardrail (answer ONLY from context) is preserved from the
original app so the assistant refuses to invent procedures.
"""

from collections.abc import Iterator

from huggingface_hub import InferenceClient

from app.config import settings
from app.retriever import RetrievedDoc, get_retriever

SYSTEM_PROMPT = """You are a Center Desk assistant for a residence hall.

Answer ONLY using the context below.
If the context does not contain the answer, say exactly: "I don't have that procedure in my knowledge base."
Do not invent steps. If you do not know the answer from the context, tell the user to reach someone on the duty chain."""


def _build_messages(context: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
    ]


def _client() -> InferenceClient:
    if not settings.hf_token:
        raise RuntimeError("HF token not set. Add HF_TOKEN to backend/.env")
    return InferenceClient(api_key=settings.hf_token)


def _messages_for(question: str) -> list[dict]:
    docs: list[RetrievedDoc] = get_retriever().search_filtered(question)
    context = "\n\n".join(d.text for d in docs)
    return _build_messages(context, question)


def answer_stream(question: str) -> Iterator[str]:
    """Yield answer tokens as they arrive (used by the streaming API). Yields
    nothing extra if no context passes the threshold — the model then states it
    lacks the info."""
    messages = _messages_for(question)
    stream = _client().chat.completions.create(
        model=settings.hf_model,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        # Final/keep-alive chunks can carry an empty choices list — skip them.
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def answer_text(question: str) -> str:
    """Return the full answer in one (non-streaming) call. Used by batch eval,
    where streaming adds flakiness without benefit."""
    messages = _messages_for(question)
    resp = _client().chat.completions.create(
        model=settings.hf_model,
        messages=messages,
        stream=False,
    )
    return (resp.choices[0].message.content or "").strip()


def retrieve_only(question: str) -> list[RetrievedDoc]:
    """Expose retrieval without generation — used by the eval harness."""
    return get_retriever().search_filtered(question)
