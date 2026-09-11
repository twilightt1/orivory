"""Tests for fallback / degradation counters (RED first)."""
from __future__ import annotations

import pytest

from app.observability.fallbacks import (
    count_fallback,
    fallback_counts,
    reset_fallback_counts,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_fallback_counts()
    yield
    reset_fallback_counts()


def test_count_increments_and_returns_total():
    assert count_fallback("retrieval.rerank_failed") == 1
    assert count_fallback("retrieval.rerank_failed") == 2


def test_paths_are_independent():
    count_fallback("a")
    count_fallback("b")
    count_fallback("a")
    assert fallback_counts() == {"a": 2, "b": 1}


def test_snapshot_is_a_copy():
    count_fallback("a")
    snapshot = fallback_counts()
    snapshot["a"] = 999
    assert fallback_counts() == {"a": 1}
