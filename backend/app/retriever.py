"""Load the FAISS index and run similarity search.

Kept as a small, single-responsibility module so it can be unit-tested in
isolation and reused by both the API and the eval harness.
"""

import pickle
from dataclasses import dataclass
from functools import lru_cache

import faiss
import numpy as np
from fastembed import TextEmbedding

from app.config import settings


@dataclass
class RetrievedDoc:
    text: str
    question: str
    answer: str
    score: float  # cosine similarity in [-1, 1]; higher is more relevant


class Retriever:
    def __init__(self) -> None:
        self._index = faiss.read_index(str(settings.index_dir / "index.faiss"))
        with open(settings.index_dir / "store.pkl", "rb") as f:
            store = pickle.load(f)
        self._metadata = store["metadata"]
        # Use the same embedding model the index was built with.
        self._embedder = TextEmbedding(model_name=store["model"])

    def search(self, query: str, k: int | None = None) -> list[RetrievedDoc]:
        k = k or settings.top_k
        # query_embed applies the model's query-side instruction (bge prepends a
        # retrieval instruction to queries, improving asymmetric search).
        vec = np.array(list(self._embedder.query_embed([query])), dtype="float32")
        vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12

        scores, idxs = self._index.search(vec, k)
        results: list[RetrievedDoc] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            results.append(
                RetrievedDoc(
                    text=meta["text"],
                    question=meta["question"],
                    answer=meta["answer"],
                    score=float(score),
                )
            )
        return results

    def search_filtered(self, query: str, k: int | None = None) -> list[RetrievedDoc]:
        """Search, then drop docs below the configured cosine threshold."""
        return [d for d in self.search(query, k) if d.score >= settings.score_threshold]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Cached singleton — the index + embedder load once per process."""
    return Retriever()
