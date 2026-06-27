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
    # the LLM is not fed loosely related context. Tuned with eval/run_eval.py:
    # every in-scope hit scores >= 0.753 while off-topic queries top out at
    # ~0.645, so 0.7 gives 0 false refusals and 100% off-topic refusal.
    score_threshold: float = 0.7

    # --- LLM (Hugging Face Inference, free tier) --------------------------
    # Accept either HF_TOKEN or the legacy API_KEY env var.
    hf_token: str | None = Field(
        default=None, validation_alias=AliasChoices("HF_TOKEN", "API_KEY")
    )
    hf_model: str = "meta-llama/Llama-3.1-8B-Instruct"

    # --- Evaluation judge (LLM-as-judge) ----------------------------------
    # The judge is independent from the generator: a stronger model grades the
    # free HF model's answers. Runs offline at eval time only, so cost is tiny.
    openai_api_key: str | None = None
    judge_model: str = "gpt-4o-mini"

    # --- API ---------------------------------------------------------------
    # Comma-separated CORS origins allowed to call this backend (the Next.js dev
    # server, then the deployed frontend URL). In production set CORS_ORIGINS to
    # your Vercel URL, e.g. CORS_ORIGINS="https://my-app.vercel.app".
    # Stored as a raw string (not list) so a plain env value needs no JSON.
    cors_origins_raw: str = Field(
        default="http://localhost:3000", validation_alias="CORS_ORIGINS"
    )

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins_raw.split(",") if o.strip()]

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
