"""
src/grading/relevance_grader.py

Uses Groq (llama-3.3-70b-versatile) to score how relevant each retrieved
chunk is to the user's question.

Each chunk gets a score 0.0–1.0 and a one-sentence reason.
The grader is the key differentiator in CRAG vs standard RAG.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from configs.prompts import RELEVANCE_GRADER_HUMAN, RELEVANCE_GRADER_SYSTEM
from src.retrieval.retriever import RetrievedChunk
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class GradedChunk:
    """A retrieved chunk annotated with a relevance score."""

    chunk: RetrievedChunk
    relevance_score: float
    reason: str

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def page_number(self) -> int:
        return self.chunk.page_number


class RelevanceGrader:
    """
    Grades each retrieved chunk for relevance to a question.

    Design choices:
    - temperature=0.0 for deterministic scoring
    - JSON-mode parsing with fallback for robustness
    - Grades chunks in sequence (could parallelise with asyncio if latency matters)
    """

    def __init__(self) -> None:
        settings = get_settings()
        # Low temp for deterministic grading; small max_tokens — just JSON output
        self._llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0.0,
            max_tokens=150,
        )

    def grade(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> list[GradedChunk]:
        """
        Score each chunk for relevance to the question.

        Returns:
            List of GradedChunk, same order as input.
        """
        graded: list[GradedChunk] = []

        for chunk in chunks:
            score, reason = self._grade_single(question, chunk.text)
            graded.append(
                GradedChunk(chunk=chunk, relevance_score=score, reason=reason)
            )
            logger.debug(
                "chunk_graded",
                score=score,
                source=chunk.source,
                page=chunk.page_number,
                reason=reason[:80],
            )

        return graded

    def _grade_single(self, question: str, document: str) -> tuple[float, str]:
        """Call the LLM to grade one chunk. Returns (score, reason)."""
        messages = [
            SystemMessage(content=RELEVANCE_GRADER_SYSTEM),
            HumanMessage(
                content=RELEVANCE_GRADER_HUMAN.format(
                    question=question,
                    document=document[:1500],  # cap context to avoid token waste
                )
            ),
        ]

        try:
            response = self._llm.invoke(messages)
            return self._parse_grade(response.content)
        except Exception as exc:
            logger.warning(
                "grading_failed",
                error=str(exc),
                fallback_score=0.5,
            )
            return 0.5, "Grading failed — defaulting to ambiguous."

    @staticmethod
    def _parse_grade(content: str) -> tuple[float, str]:
        """
        Parse LLM JSON response. Graceful fallback if malformed.
        Handles both clean JSON and JSON wrapped in markdown fences.
        """
        text = content.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                l for l in lines if not l.startswith("```")
            ).strip()

        try:
            data = json.loads(text)
            score = float(data.get("score", 0.5))
            score = max(0.0, min(1.0, score))
            reason = str(data.get("reason", "")).strip()
            return score, reason
        except (json.JSONDecodeError, KeyError, ValueError):
            # Last-resort: try to extract a float from the response
            import re
            match = re.search(r'"score"\s*:\s*([0-9.]+)', text)
            if match:
                return float(match.group(1)), "Parsed from malformed JSON."
            return 0.5, "Could not parse grade response."
