"""P2.1 tests: salience feedback loop (bump + decay)."""
from __future__ import annotations

import pytest

from app.retrieval.memory.salience import DEFAULT_BUMP_STEP, SALIENCE_MAX, next_salience

pytestmark = pytest.mark.rag


class TestNextSalience:
    def test_increments_toward_one(self):
        assert next_salience(0.5) == 0.55
        assert next_salience(0.0) == round(DEFAULT_BUMP_STEP, 6)

    def test_asymptotic_never_exceeds_max(self):
        s = 0.5
        for _ in range(1000):
            s = next_salience(s)
        assert s <= SALIENCE_MAX
        assert s > 0.99  # converges near 1.0

    def test_clamps_out_of_range_input(self):
        assert next_salience(1.5) == SALIENCE_MAX  # clamped down first
        assert next_salience(-1.0) == round(DEFAULT_BUMP_STEP, 6)

    def test_diminishing_returns(self):
        # The increment shrinks as salience rises.
        low_gain = next_salience(0.1) - 0.1
        high_gain = next_salience(0.9) - 0.9
        assert low_gain > high_gain


