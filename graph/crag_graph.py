"""
graph/crag_graph.py

The corrective RAG state machine built with LangGraph.

Node execution order:
  retrieve → grade → [conditional] → generate → END

Conditional routing:
  high      → generate (local docs)
  low       → rewrite_query → web_search → generate (web only)
  ambiguous → web_search → fuse_and_generate

Each node receives the full CRAGState dict and returns a partial update.
"""

from __future__ import annotations

import time
import uuid

from langgraph.graph import StateGraph, END

from graph.state import CRAGState
from src.retrieval.retriever import Retriever
from src.grading.relevance_grader import RelevanceGrader
from src.generation.answer_generator import AnswerGenerator
from src.web_search.tavily_search import QueryRewriter, TavilySearcher
from src.utils.config import get_settings
from src.utils.audit_logger import get_audit_logger
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Singletons (lazy init inside each node avoids import-time side effects) ──
_retriever: Retriever | None = None
_grader: RelevanceGrader | None = None
_generator: AnswerGenerator | None = None
_rewriter: QueryRewriter | None = None
_searcher: TavilySearcher | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _get_grader() -> RelevanceGrader:
    global _grader
    if _grader is None:
        _grader = RelevanceGrader()
    return _grader


def _get_generator() -> AnswerGenerator:
    global _generator
    if _generator is None:
        _generator = AnswerGenerator()
    return _generator


def _get_rewriter() -> QueryRewriter:
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter()
    return _rewriter


def _get_searcher() -> TavilySearcher:
    global _searcher
    if _searcher is None:
        _searcher = TavilySearcher()
    return _searcher


# ── Nodes ────────────────────────────────────────────────────────────────────


def node_retrieve(state: CRAGState) -> dict:
    """Retrieve top-k chunks from ChromaDB."""
    logger.info("node_retrieve", question=state["question"][:80])
    chunks = _get_retriever().retrieve(state["question"])
    return {"retrieved_chunks": chunks}


def node_grade(state: CRAGState) -> dict:
    """Grade each chunk and compute confidence level."""
    settings = get_settings()
    graded = _get_grader().grade(state["question"], state["retrieved_chunks"])

    if not graded:
        avg = 0.0
        confidence = "low"
    else:
        avg = round(sum(c.relevance_score for c in graded) / len(graded), 4)
        if avg >= settings.high_conf_threshold:
            confidence = "high"
        elif avg < settings.low_conf_threshold:
            confidence = "low"
        else:
            confidence = "ambiguous"

    logger.info(
        "node_grade",
        avg_score=avg,
        confidence=confidence,
        num_chunks=len(graded),
    )
    return {
        "graded_chunks": graded,
        "avg_score": avg,
        "confidence": confidence,
    }


def node_rewrite_query(state: CRAGState) -> dict:
    """Rewrite the query for better web search retrieval."""
    rewritten = _get_rewriter().rewrite(state["question"])
    return {"web_query": rewritten}


def node_web_search(state: CRAGState) -> dict:
    """Perform Tavily web search using the (possibly rewritten) query."""
    query = state.get("web_query") or state["question"]
    results = _get_searcher().search(query)
    return {"web_results": results}


def node_generate_high(state: CRAGState) -> dict:
    """HIGH confidence: generate answer from local docs only."""
    result = _get_generator().generate_from_local(
        question=state["question"],
        graded_chunks=state["graded_chunks"],
        confidence="high",
        avg_score=state["avg_score"],
    )
    _write_audit(state, result, latency_start=state.get("_start_time", time.time()))
    return {"result": result}


def node_generate_low(state: CRAGState) -> dict:
    """LOW confidence: generate answer from web results only."""
    result = _get_generator().generate_from_web(
        question=state["question"],
        web_results=state.get("web_results", []),
        web_query=state.get("web_query", state["question"]),
        avg_score=state["avg_score"],
    )
    _write_audit(state, result, latency_start=state.get("_start_time", time.time()))
    return {"result": result}


def node_generate_fused(state: CRAGState) -> dict:
    """AMBIGUOUS: fuse local and web results."""
    result = _get_generator().generate_fused(
        question=state["question"],
        graded_chunks=state["graded_chunks"],
        web_results=state.get("web_results", []),
        web_query=state.get("web_query", state["question"]),
        avg_score=state["avg_score"],
    )
    _write_audit(state, result, latency_start=state.get("_start_time", time.time()))
    return {"result": result}


# ── Routing ──────────────────────────────────────────────────────────────────


def route_after_grade(state: CRAGState) -> str:
    """Decide which node to execute based on confidence."""
    confidence = state.get("confidence", "low")
    route_map = {
        "high": "generate_high",
        "ambiguous": "rewrite_query_ambiguous",
        "low": "rewrite_query_low",
    }
    return route_map.get(confidence, "rewrite_query_low")


# ── Audit helper ─────────────────────────────────────────────────────────────


def _write_audit(state: CRAGState, result, latency_start: float) -> None:
    try:
        audit = get_audit_logger()
        chunk_scores = [
            {
                "text_preview": c.text[:80],
                "score": c.relevance_score,
                "source": c.source,
            }
            for c in state.get("graded_chunks", [])
        ]
        audit.log(
            {
                "request_id": state.get("request_id", str(uuid.uuid4())),
                "question": state["question"],
                "confidence": state.get("confidence"),
                "avg_score": state.get("avg_score"),
                "chunk_scores": chunk_scores,
                "web_triggered": result.web_triggered,
                "web_query": result.web_query,
                "sources_used": result.sources_used,
                "latency_ms": round((time.time() - latency_start) * 1000),
            }
        )
    except Exception as exc:
        logger.warning("audit_log_failed", error=str(exc))


# ── Graph assembly ────────────────────────────────────────────────────────────


def build_crag_graph() -> StateGraph:
    graph = StateGraph(CRAGState)

    graph.add_node("retrieve", node_retrieve)
    graph.add_node("grade", node_grade)
    graph.add_node("rewrite_query_low", node_rewrite_query)
    graph.add_node("rewrite_query_ambiguous", node_rewrite_query)
    graph.add_node("web_search_low", node_web_search)
    graph.add_node("web_search_ambiguous", node_web_search)
    graph.add_node("generate_high", node_generate_high)
    graph.add_node("generate_low", node_generate_low)
    graph.add_node("generate_fused", node_generate_fused)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "generate_high": "generate_high",
            "rewrite_query_low": "rewrite_query_low",
            "rewrite_query_ambiguous": "rewrite_query_ambiguous",
        },
    )

    graph.add_edge("rewrite_query_low", "web_search_low")
    graph.add_edge("rewrite_query_ambiguous", "web_search_ambiguous")
    graph.add_edge("web_search_low", "generate_low")
    graph.add_edge("web_search_ambiguous", "generate_fused")

    graph.add_edge("generate_high", END)
    graph.add_edge("generate_low", END)
    graph.add_edge("generate_fused", END)

    return graph.compile()


# Compiled graph — import this in the API and CLI
crag_graph = build_crag_graph()
