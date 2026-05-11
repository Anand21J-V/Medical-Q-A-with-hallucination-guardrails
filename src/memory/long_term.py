"""
src/memory/long_term.py

Long-Term Memory: extracts and persists durable facts about the user
across all sessions. Facts are extracted by a small LLM call after each answer.
"""
from __future__ import annotations
import json
from groq import Groq
from src.memory.supabase_client import get_supabase
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_EXTRACT_PROMPT = """\
You are a medical memory extractor. Given a user question and the AI answer,
extract any durable personal facts about the user (conditions, medications,
allergies, preferences). Return a JSON array of objects, each with:
  - "fact": short declarative sentence (e.g. "User has hypertension")
  - "category": one of ["condition", "medication", "allergy", "preference", "other"]

If no durable facts are found, return [].
Respond ONLY with valid JSON. No explanation.

Question: {question}
Answer: {answer}
"""

class LongTermMemory:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self._db = get_supabase()
        self._settings = get_settings()

    def get_facts(self) -> list[dict]:
        """Retrieve all stored facts for this user."""
        try:
            res = (
                self._db.table("long_term_memory")
                .select("fact, category, source_question")
                .eq("user_id", self.user_id)
                .order("updated_at", desc=True)
                .execute()
            )
            return res.data or []
        except Exception as e:
            logger.warning("ltm_fetch_failed", error=str(e))
            return []

    def format_for_prompt(self) -> str:
        """Format stored facts as a string to inject into the system prompt."""
        facts = self.get_facts()
        if not facts:
            return ""
        lines = ["## Known facts about this user:"]
        for f in facts:
            lines.append(f"- [{f['category']}] {f['fact']}")
        return "\n".join(lines)

    def extract_and_save(self, question: str, answer: str, source_question: str) -> None:
        """Call LLM to extract facts, then upsert them into Supabase."""
        try:
            client = Groq(api_key=self._settings.groq_api_key)
            resp = client.chat.completions.create(
                model=self._settings.groq_model,
                temperature=0.0,
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": _EXTRACT_PROMPT.format(question=question, answer=answer)
                }]
            )
            raw = resp.choices[0].message.content.strip()
            facts = json.loads(raw)
            for item in facts:
                if not item.get("fact"):
                    continue
                self._db.table("long_term_memory").upsert({
                    "user_id": self.user_id,
                    "fact": item["fact"],
                    "category": item.get("category", "other"),
                    "source_question": source_question,
                }, on_conflict="user_id,fact").execute()
            if facts:
                logger.info("ltm_facts_saved", count=len(facts), user_id=self.user_id)
        except Exception as e:
            logger.warning("ltm_extract_failed", error=str(e))