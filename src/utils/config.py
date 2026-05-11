"""
src/utils/config.py

Central configuration via Pydantic Settings.
All values can be overridden by environment variables or a .env file.
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── LLM ──────────────────────────────────────────────────────────────
    groq_api_key: str = Field(..., description="Groq API key")
    groq_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Groq model identifier",
    )
    groq_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    groq_max_tokens: int = Field(default=1024, gt=0)

    # ── Web Search ────────────────────────────────────────────────────────
    tavily_api_key: str = Field(..., description="Tavily API key")
    tavily_max_results: int = Field(default=5, gt=0)

    # ── Vector Store ──────────────────────────────────────────────────────
    chroma_path: str = Field(default="./data/vectorstore")
    collection_name: str = Field(default="medical_docs")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ── Retrieval & Chunking ──────────────────────────────────────────────
    top_k: int = Field(default=5, gt=0, le=20)
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)

    # ── CRAG Decision Thresholds ──────────────────────────────────────────
    high_conf_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    low_conf_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    # ── API ───────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=1)

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    audit_log_path: str = Field(default="./logs/crag_audit.jsonl")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
