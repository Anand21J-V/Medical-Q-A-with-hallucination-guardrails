"""
src/generation/answer_generator.py

Synthesises the final answer from graded local documents, web results, or both.
Handles all three CRAG paths: HIGH (local only), LOW (web only), AMBIGUOUS (fusion).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from configs.prompts import (
    ANSWER_GENERATOR_HUMAN,
    ANSWER_GENERATOR_SYSTEM,
    FUSION_HUMAN,
    FUSION_SYSTEM,
)
from src.grading.relevance_grader import GradedChunk
from src.web_search.tavily_search import WebResult
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class GeneratedAnswer:
    """The complete answer with provenance information."""

    answer: str
    confidence: str                  # "high" | "ambiguous" | "low"
    avg_relevance_score: float
    sources_used: list[str]
    web_triggered: bool
    web_query: str | None


class AnswerGenerator:
    """
    Generates answers using Groq.
    Selects the appropriate prompt path based on CRAG confidence level.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=settings.groq_temperature,
            max_tokens=settings.groq_max_tokens,
        )

    def generate_from_local(
        self,
        question: str,
        graded_chunks: Sequence[GradedChunk],
        confidence: str,
        avg_score: float,
    ) -> GeneratedAnswer:
        """HIGH confidence path: answer entirely from local documents."""
        sources_block = self._format_local_sources(graded_chunks)
        answer = self._call_llm(
            system=ANSWER_GENERATOR_SYSTEM,
            human=ANSWER_GENERATOR_HUMAN.format(
                question=question, sources=sources_block
            ),
        )
        return GeneratedAnswer(
            answer=answer,
            confidence=confidence,
            avg_relevance_score=avg_score,
            sources_used=self._extract_source_refs(graded_chunks),
            web_triggered=False,
            web_query=None,
        )

    def generate_from_web(
        self,
        question: str,
        web_results: Sequence[WebResult],
        web_query: str,
        avg_score: float,
    ) -> GeneratedAnswer:
        """LOW confidence path: answer from web results only."""
        sources_block = self._format_web_sources(web_results)
        answer = self._call_llm(
            system=ANSWER_GENERATOR_SYSTEM,
            human=ANSWER_GENERATOR_HUMAN.format(
                question=question, sources=sources_block
            ),
        )
        return GeneratedAnswer(
            answer=answer,
            confidence="low",
            avg_relevance_score=avg_score,
            sources_used=[r.url for r in web_results],
            web_triggered=True,
            web_query=web_query,
        )

    def generate_fused(
        self,
        question: str,
        graded_chunks: Sequence[GradedChunk],
        web_results: Sequence[WebResult],
        web_query: str,
        avg_score: float,
    ) -> GeneratedAnswer:
        """AMBIGUOUS path: fuse local docs and web results."""
        local_block = self._format_local_sources(graded_chunks, prefix="L")
        web_block = self._format_web_sources(web_results, prefix="W")

        answer = self._call_llm(
            system=FUSION_SYSTEM,
            human=FUSION_HUMAN.format(
                question=question,
                local_sources=local_block,
                web_sources=web_block,
            ),
        )
        all_sources = (
            self._extract_source_refs(graded_chunks)
            + [r.url for r in web_results]
        )
        return GeneratedAnswer(
            answer=answer,
            confidence="ambiguous",
            avg_relevance_score=avg_score,
            sources_used=all_sources,
            web_triggered=True,
            web_query=web_query,
        )

    # ── Private helpers ───────────────────────────────────────────────────

    def _call_llm(self, system: str, human: str) -> str:
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=human),
        ]
        try:
            response = self._llm.invoke(messages)
            return response.content.strip()
        except Exception as exc:
            logger.error("answer_generation_failed", error=str(exc))
            return (
                "An error occurred while generating the answer. "
                "Please try again."
            )

    @staticmethod
    def _format_local_sources(
        chunks: Sequence[GradedChunk], prefix: str = ""
    ) -> str:
        lines = []
        for i, c in enumerate(chunks, start=1):
            label = f"[{prefix}{i}]" if prefix else f"[{i}]"
            lines.append(
                f"{label} Source: {c.source} (page {c.page_number})\n"
                f"Relevance: {c.relevance_score:.2f}\n"
                f"{c.text}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_web_sources(
        results: Sequence[WebResult], prefix: str = ""
    ) -> str:
        lines = []
        for i, r in enumerate(results, start=1):
            label = f"[{prefix}{i}]" if prefix else f"[{i}]"
            lines.append(
                f"{label} {r.title}\n"
                f"URL: {r.url}\n"
                f"Relevance: {r.score:.2f}\n"
                f"{r.content[:800]}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _extract_source_refs(chunks: Sequence[GradedChunk]) -> list[str]:
        seen: set[str] = set()
        refs = []
        for c in chunks:
            ref = f"{c.source} (p.{c.page_number})"
            if ref not in seen:
                refs.append(ref)
                seen.add(ref)
        return refs
