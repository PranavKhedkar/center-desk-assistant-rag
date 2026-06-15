"""Typed application configuration.

Centralizes every tunable knob (model names, paths, retrieval params) in one
place instead of scattering os.getenv calls across the codebase. Values can be
overridden via environment variables or a .env file. This is the production
pattern: configuration is explicit, typed, and validated at startup.
"""

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve paths relative to the backend/ directory so the app works regardless
# of the current working directory it is launched from.
BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # --- Data / index paths ------------------------------------------------
    csv_path: Path = BACKEND_DIR / "data" / "Center_Desk_Manual.csv"
    index_dir: Path = BACKEND_DIR / "vector_store"

    # --- Embeddings (local, free) -----------------------------------------
    # bge-small is a strong, small (384-dim) English embedding model that runs
    # locally via fastembed's ONNX runtime — no per-query API cost, no PyTorch.
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- Retrieval ---------------------------------------------------------
    top_k: int = 3
    # Cosine-similarity floor: retrieved docs scoring below this are dropped so
    # the LLM is not fed loosely related context. Chosen empirically — on-topic
    # queries score 0.75+ while off-topic ones top out near 0.50, so 0.6 leaves
    # a clean margin. The eval harness should re-confirm this.
    score_threshold: float = 0.6

    # --- LLM (Hugging Face Inference, free tier) --------------------------
    # Accept either HF_TOKEN or the legacy API_KEY env var.
    hf_token: str | None = Field(
        default=None, validation_alias=AliasChoices("HF_TOKEN", "API_KEY")
    )
    hf_model: str = "meta-llama/Llama-3.1-8B-Instruct"

    # --- API ---------------------------------------------------------------
    # CORS origins allowed to call this backend (the Next.js dev server, then
    # the deployed frontend URL).
    cors_origins: list[str] = ["http://localhost:3000"]

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
