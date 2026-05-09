"""
tests/test_grader.py

Unit tests for RelevanceGrader._parse_grade — no LLM calls needed.
Integration test (marked slow) hits Groq if GROQ_API_KEY is set.
"""

import pytest
from src.grading.relevance_grader import RelevanceGrader


class TestParseGrade:
    def test_valid_json(self):
        score, reason = RelevanceGrader._parse_grade(
            '{"score": 0.85, "reason": "Directly addresses the question."}'
        )
        assert score == pytest.approx(0.85)
        assert "Directly" in reason

    def test_score_clamped_high(self):
        score, _ = RelevanceGrader._parse_grade('{"score": 1.5, "reason": "x"}')
        assert score == 1.0

    def test_score_clamped_low(self):
        score, _ = RelevanceGrader._parse_grade('{"score": -0.3, "reason": "x"}')
        assert score == 0.0

    def test_markdown_fenced_json(self):
        raw = '```json\n{"score": 0.6, "reason": "Partial match."}\n```'
        score, reason = RelevanceGrader._parse_grade(raw)
        assert score == pytest.approx(0.6)

    def test_malformed_json_regex_fallback(self):
        score, _ = RelevanceGrader._parse_grade(
            'Here is my assessment: "score": 0.75 and stuff'
        )
        assert score == pytest.approx(0.75)

    def test_completely_invalid_returns_default(self):
        score, _ = RelevanceGrader._parse_grade("I cannot grade this.")
        assert score == 0.5
