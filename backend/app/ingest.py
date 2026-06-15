"""Build the FAISS vector index from the Q&A CSV.

Mirrors the original design decision: we embed the *question* as the retrieval
key, but store the full "Question / Answer" text as the document handed to the
LLM. The difference from before is that embeddings are now produced locally with
fastembed (free, no API calls) and the index uses cosine similarity so the
relevance threshold is interpretable.

Run:  python -m app.ingest
"""

import pickle

import faiss
import numpy as np
import pandas as pd
from fastembed import TextEmbedding

from app.config import settings


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows so inner-product search == cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return vectors / norms


def build_index() -> None:
    df = pd.read_csv(settings.csv_path)
    questions = df["input_text"].astype(str).tolist()
    answers = df["target_text"].astype(str).tolist()

    # Stored document = full Q+A so the LLM sees the canonical answer text.
    documents = [f"Question: {q}\nAnswer: {a}" for q, a in zip(questions, answers)]

    embedder = TextEmbedding(model_name=settings.embedding_model)
    # passage_embed is the document-side embedding; we embed the questions as the
    # retrieval keys (matching the original retrieve-on-question design).
    vectors = np.array(list(embedder.passage_embed(questions)), dtype="float32")
    vectors = _normalize(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine
    index.add(vectors)

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(settings.index_dir / "index.faiss"))

    # Persist the documents + raw fields alongside the index so the retriever can
    # map a FAISS row back to its text and metadata.
    metadata = [
        {"text": doc, "question": q, "answer": a}
        for doc, q, a in zip(documents, questions, answers)
    ]
    with open(settings.index_dir / "store.pkl", "wb") as f:
        pickle.dump({"metadata": metadata, "model": settings.embedding_model}, f)

    print(f"Indexed {len(documents)} entries (dim={dim}) -> {settings.index_dir}")


if __name__ == "__main__":
    build_index()
