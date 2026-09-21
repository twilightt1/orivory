"""P4 hardening tests: context budget, query limits."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.rag


class TestChatRequestLimits:
    def test_query_max_length_enforced(self):
        from pydantic import ValidationError

        from app.schemas.conversation import ChatRequest

        with pytest.raises(ValidationError):
            ChatRequest(query="x" * 10_001)

    def test_query_within_limit_ok(self):
        from app.schemas.conversation import ChatRequest

        req = ChatRequest(query="x" * 10_000)
        assert len(req.query) == 10_000
