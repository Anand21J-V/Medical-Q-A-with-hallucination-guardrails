"""
src/memory/short_term.py

Short-Term Memory: stores and retrieves the last N turns of a session.
Backed by Supabase `conversation_turns` table.
"""
from __future__ import annotations
from dataclasses import dataclass
from src.memory.supabase_client import get_supabase
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ConversationTurn:
    question: str
    answer: str
    confidence: str
    sources_used: list[str]

class ShortTermMemory:
    def __init__(self, session_id: str, user_id: str, max_turns: int = 6):
        self.session_id = session_id
        self.user_id = user_id
        self.max_turns = max_turns  # last N turns to inject into context
        self._db = get_supabase()

    def get_history(self) -> list[ConversationTurn]:
        """Fetch the last max_turns turns for this session."""
        try:
            res = (
                self._db.table("conversation_turns")
                .select("question, answer, confidence, sources_used")
                .eq("session_id", self.session_id)
                .order("created_at", desc=True)
                .limit(self.max_turns)
                .execute()
            )
            # Reverse so oldest first (natural reading order)
            turns = list(reversed(res.data or []))
            return [
                ConversationTurn(
                    question=t["question"],
                    answer=t["answer"],
                    confidence=t["confidence"],
                    sources_used=t.get("sources_used") or [],
                )
                for t in turns
            ]
        except Exception as e:
            logger.warning("stm_fetch_failed", error=str(e))
            return []

    def format_for_prompt(self) -> str:
        """Format history as a string to prepend to the system prompt."""
        turns = self.get_history()
        if not turns:
            return ""
        lines = ["## Previous conversation context:"]
        for i, t in enumerate(turns, 1):
            lines.append(f"Q{i}: {t.question}")
            lines.append(f"A{i}: {t.answer}")
        return "\n".join(lines)

    def save_turn(
        self,
        request_id: str,
        question: str,
        answer: str,
        confidence: str,
        avg_relevance_score: float,
        web_triggered: bool,
        sources_used: list[str],
    ) -> None:
        """Persist a completed turn to Supabase."""
        try:
            self._db.table("conversation_turns").insert({
                "session_id": self.session_id,
                "user_id": self.user_id,
                "request_id": request_id,
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "avg_relevance_score": avg_relevance_score,
                "web_triggered": web_triggered,
                "sources_used": sources_used,
            }).execute()
        except Exception as e:
            logger.warning("stm_save_failed", error=str(e))