"""
src/api/schemas.py

Pydantic v2 models for all API request and response bodies.
Using strict typing so invalid inputs are caught at the boundary.
"""

from __future__ import annotations

from typing import Annotated
from pydantic import BaseModel, Field


# ── /ask ──────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: Annotated[str, Field(min_length=5, max_length=1000)]
    top_k: Annotated[int, Field(default=5, ge=1, le=20)] = 5
    # Memory fields — both are optional.
    # Omit session_id to start a fresh conversation; include it to continue one.
    # Omit user_id to use anonymous mode (no Long-Term Memory personalisation).
    session_id: str | None = Field(
        default=None,
        description="Opaque session token. Include to continue a conversation.",
    )
    user_id: str | None = Field(
        default=None,
        description="Stable user identifier. Required for Long-Term Memory.",
    )


class ChunkTrace(BaseModel):
    text_preview: str
    source: str
    page_number: int
    relevance_score: float


class AskResponse(BaseModel):
    request_id: str
    session_id: str          # Always returned — mint and store if you didn't send one
    question: str
    answer: str
    confidence: str          # "high" | "ambiguous" | "low"
    avg_relevance_score: float
    web_triggered: bool
    web_query: str | None
    sources_used: list[str]
    crag_trace: list[ChunkTrace]
    latency_ms: int


# ── /ingest ───────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    filename: str
    pages_loaded: int
    chunks_created: int
    chunks_upserted: int
    collection_size: int


# ── /health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    collection_name: str
    collection_size: int
    groq_model: str


# ── /audit ────────────────────────────────────────────────────────────────────

class AuditResponse(BaseModel):
    records: list[dict]
    total: int