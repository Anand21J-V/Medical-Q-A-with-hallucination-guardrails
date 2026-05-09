"""
graph/state.py

The single TypedDict that flows through every node of the LangGraph state machine.
Using TypedDict (not Pydantic) is idiomatic for LangGraph.
"""

from __future__ import annotations

from typing import TypedDict

from src.grading.relevance_grader import GradedChunk
from src.retrieval.retriever import RetrievedChunk
from src.web_search.tavily_search import WebResult
from src.generation.answer_generator import GeneratedAnswer


class CRAGState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────
    question: str
    request_id: str

    # ── Retrieval ─────────────────────────────────────────────────────────
    retrieved_chunks: list[RetrievedChunk]

    # ── Grading ───────────────────────────────────────────────────────────
    graded_chunks: list[GradedChunk]
    avg_score: float
    confidence: str  # "high" | "ambiguous" | "low"

    # ── Web search ────────────────────────────────────────────────────────
    web_query: str | None
    web_results: list[WebResult]

    # ── Output ────────────────────────────────────────────────────────────
    result: GeneratedAnswer | None
    error: str | None
