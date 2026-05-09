"""
tests/test_graph.py

Tests for the CRAG graph routing logic — no external API calls.
"""

import pytest
from graph.crag_graph import route_after_grade
from graph.state import CRAGState


class TestRouteAfterGrade:
    def test_high_confidence_routes_to_generate(self):
        state: CRAGState = {
            "question": "What is aspirin?",
            "confidence": "high",
            "avg_score": 0.85,
        }
        assert route_after_grade(state) == "generate_high"

    def test_low_confidence_routes_to_web(self):
        state: CRAGState = {
            "question": "What is aspirin?",
            "confidence": "low",
            "avg_score": 0.2,
        }
        assert route_after_grade(state) == "rewrite_query_low"

    def test_ambiguous_confidence_routes_to_fusion(self):
        state: CRAGState = {
            "question": "What is aspirin?",
            "confidence": "ambiguous",
            "avg_score": 0.55,
        }
        assert route_after_grade(state) == "rewrite_query_ambiguous"

    def test_missing_confidence_defaults_to_low(self):
        state: CRAGState = {"question": "test"}
        assert route_after_grade(state) == "rewrite_query_low"
