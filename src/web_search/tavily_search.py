"""
src/web_search/tavily_search.py

Two responsibilities:
1. QueryRewriter  — rewrites low-quality or ambiguous queries using Groq
2. TavilySearcher — performs web search via Tavily API and returns structured results
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from tavily import TavilyClient

from configs.prompts import QUERY_REWRITER_HUMAN, QUERY_REWRITER_SYSTEM
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class WebResult:
    """A single web search result."""

    title: str
    url: str
    content: str
    score: float  # Tavily relevance score 0–1


class QueryRewriter:
    """
    Rewrites the user's question into a better search query.
    Used in the LOW confidence path before hitting Tavily.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0.3,
            max_tokens=80,
        )

    def rewrite(self, question: str) -> str:
        """Return a rewritten search query optimised for web retrieval."""
        messages = [
            SystemMessage(content=QUERY_REWRITER_SYSTEM),
            HumanMessage(
                content=QUERY_REWRITER_HUMAN.format(question=question)
            ),
        ]
        try:
            response = self._llm.invoke(messages)
            rewritten = response.content.strip().strip('"')
            logger.info(
                "query_rewritten",
                original=question[:80],
                rewritten=rewritten[:80],
            )
            return rewritten
        except Exception as exc:
            logger.warning("query_rewrite_failed", error=str(exc))
            return question  # fall back to original


class TavilySearcher:
    """
    Wraps the Tavily client with structured return types.
    Filters out low-score results (< 0.3) to reduce noise.
    """

    _MIN_SCORE = 0.3

    def __init__(self) -> None:
        settings = get_settings()
        self._client = TavilyClient(api_key=settings.tavily_api_key)
        self._max_results = settings.tavily_max_results

    def search(self, query: str) -> list[WebResult]:
        """
        Search the web and return filtered, structured results.

        Args:
            query: The search query (ideally rewritten by QueryRewriter).

        Returns:
            List of WebResult sorted by Tavily score descending.
        """
        try:
            response = self._client.search(
                query=query,
                search_depth="basic",
                max_results=self._max_results,
                include_answer=False,
            )
        except Exception as exc:
            logger.error("tavily_search_failed", error=str(exc), query=query)
            return []

        results: list[WebResult] = []
        for item in response.get("results", []):
            score = float(item.get("score", 0.0))
            if score < self._MIN_SCORE:
                continue
            results.append(
                WebResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=score,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        logger.info(
            "web_search_complete",
            query=query[:80],
            total_results=len(results),
        )
        return results
