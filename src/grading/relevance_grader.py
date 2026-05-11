"""
src/grading/relevance_grader.py
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    All chunks are graded in PARALLEL using a thread pool — this is the
    single biggest latency improvement vs sequential grading.
    Uses llama-3.1-8b-instant (fast, cheap) instead of 70b for grading.
    """

    _MAX_WORKERS = 7  # one thread per chunk, matches TOP_K

    def __init__(self) -> None:
        settings = get_settings()
        # ── Use the fast 8b model for grading — it's equally accurate
        #    for simple JSON relevance scoring, 3-4x faster than 70b
        self._llm = ChatGroq(
            api_key=settings.groq_api_key,
            model="llama-3.1-8b-instant",   # fast grading model
            temperature=0.0,
            max_tokens=120,
        )

    def grade(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> list[GradedChunk]:
        """
        Score all chunks in parallel using a thread pool.
        Returns list in the same order as input.
        """
        if not chunks:
            return []

        results: dict[int, GradedChunk] = {}

        with ThreadPoolExecutor(max_workers=self._MAX_WORKERS) as executor:
            future_to_idx = {
                executor.submit(self._grade_single, question, chunk.text): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                chunk = chunks[idx]
                try:
                    score, reason = future.result()
                except Exception as exc:
                    logger.warning("grading_failed", error=str(exc))
                    score, reason = 0.5, "Grading failed — defaulting to ambiguous."

                results[idx] = GradedChunk(
                    chunk=chunk,
                    relevance_score=score,
                    reason=reason,
                )
                logger.debug(
                    "chunk_graded",
                    score=score,
                    source=chunk.source,
                    page=chunk.page_number,
                )

        # Return in original order
        return [results[i] for i in range(len(chunks))]

    def _grade_single(self, question: str, document: str) -> tuple[float, str]:
        messages = [
            SystemMessage(content=RELEVANCE_GRADER_SYSTEM),
            HumanMessage(
                content=RELEVANCE_GRADER_HUMAN.format(
                    question=question,
                    document=document[:1200],
                )
            ),
        ]
        try:
            response = self._llm.invoke(messages)
            return self._parse_grade(response.content)
        except Exception as exc:
            logger.warning("grading_failed", error=str(exc), fallback_score=0.5)
            return 0.5, "Grading failed — defaulting to ambiguous."

    @staticmethod
    def _parse_grade(content: str) -> tuple[float, str]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(l for l in lines if not l.startswith("```")).strip()
        try:
            data = json.loads(text)
            score = float(data.get("score", 0.5))
            score = max(0.0, min(1.0, score))
            reason = str(data.get("reason", "")).strip()
            return score, reason
        except (json.JSONDecodeError, KeyError, ValueError):
            match = re.search(r'"score"\s*:\s*([0-9.]+)', text)
            if match:
                return float(match.group(1)), "Parsed from malformed JSON."
            return 0.5, "Could not parse grade response."