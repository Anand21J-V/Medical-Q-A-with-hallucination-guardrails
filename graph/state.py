"""
graph/state.py

The single TypedDict that flows through every node of the LangGraph state machine.
Using TypedDict (not Pydantic) is idiomatic for LangGraph.
"""

from __future__ import annotations

from typing import TypedDict

from src.generation.answer_generator import GeneratedAnswer
from src.grading.relevance_grader import GradedChunk
from src.retrieval.retriever import RetrievedChunk
from src.web_search.tavily_search import WebResult


class CRAGState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────
    question: str
    request_id: str

    # ── Memory ────────────────────────────────────────────────────────────
    # Populated by the /ask route BEFORE the graph runs.
    # Consumed by AnswerGenerator nodes to personalise the system prompt.
    session_id: str          # Ties this run to a Supabase conversation
    user_id: str             # Owner of the Long-Term Memory facts
    stm_context: str         # Formatted recent turns for prompt injection
    ltm_context: str         # Formatted durable user facts for prompt injection

    # ── Retrieval ─────────────────────────────────────────────────────────
    retrieved_chunks: list[RetrievedChunk]

    # ── Grading ───────────────────────────────────────────────────────────
    graded_chunks: list[GradedChunk]
    avg_score: float
    confidence: str          # "high" | "ambiguous" | "low"

    # ── Web search ────────────────────────────────────────────────────────
    web_query: str | None
    web_results: list[WebResult]

    # ── Output ────────────────────────────────────────────────────────────
    result: GeneratedAnswer | None
    error: str | None

    # ── Internal ──────────────────────────────────────────────────────────
    _start_time: float       # Unix timestamp, used to compute latency_ms