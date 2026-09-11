"""P3 tests: quality trend aggregation."""
from __future__ import annotations

import pytest

from app.services.quality_service import aggregate_traces

pytestmark = pytest.mark.rag



class TestAggregateTraces:
    def test_empty(self):
        agg = aggregate_traces([])
        assert agg["sample_size"] == 0
        assert agg["citation_rate"] == 0.0

    def test_rates_use_own_denominators(self):
        traces = [
            # answer with sources, cited, grounded
            {
                "citation": {"required": True, "has_citation": True, "source_count": 2},
                "hallucination": {"grounded": True},
            },
            # answer with sources, NOT cited, NOT grounded
            {
                "citation": {"source_count": 1, "has_citation": False},
                "hallucination": {"grounded": False},
            },
            # chitchat: no citation/grounding signals → excluded from those rates
            {"router": {"intent": "chitchat"}},
        ]
        agg = aggregate_traces(traces)
        assert agg["sample_size"] == 3
        # citation rate over the 2 answers that had sources → 1/2
        assert agg["citation_rate"] == 0.5
        assert agg["citation_sample"] == 2
        # grounded over the 2 with a verdict → 1/2; flag rate is the complement
        assert agg["grounded_rate"] == 0.5
        assert agg["hallucination_flag_rate"] == 0.5
        assert agg["grounded_sample"] == 2

    def test_self_correction_rate_over_all(self):
        traces = [
            {"correction": [{"reason": "irrelevant_context"}]},
            {},
            {},
            {},
        ]
        agg = aggregate_traces(traces)
        assert agg["self_correction_rate"] == 0.25  # 1 of 4

    def test_confidence_excludes_not_applicable(self):
        traces = [
            {"grounding": {"label": "high", "score": 0.9}},
            {"grounding": {"label": "not_applicable", "score": 0.0}},  # excluded
            {"grounding": {"label": "low", "score": 0.3}},
        ]
        agg = aggregate_traces(traces)
        # average over the 2 applicable scores
        assert agg["avg_grounding_confidence"] == 0.6
        assert agg["grounding_confidence_sample"] == 2

    def test_latency_average(self):
        traces = [
            {"answer": {"latency_ms": 400}},
            {"answer": {"latency_ms": 600}},
            {"answer": {}},  # no latency → excluded
        ]
        agg = aggregate_traces(traces)
        assert agg["avg_answer_latency_ms"] == 500.0
        assert agg["latency_sample"] == 2

    def test_tolerates_malformed_traces(self):
        agg = aggregate_traces([None, "not a dict", 123, {"citation": "bad"}])
        assert agg["sample_size"] == 4  # does not raise
