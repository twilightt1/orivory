"""Regression test: refresh endpoint must not mutate other users' cards.

Found in full-repo review (HIGH, security): refresh_insights_endpoint applied
LLM-returned insight IDs via `db.get(InsightCard, id)` with no ownership
check — a hallucinated or echoed foreign ID let one user expire/dismiss/boost
another user's InsightCard. Every other handler in the file checks
`card.user_id != current_user.id`.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api.v1 import insights as insights_router
from app.models.insight import InsightStatusEnum


class _Scalars:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return list(self._rows)


class _Result:
    def __init__(self, rows=()):
        self._rows = rows

    def scalars(self):
        return _Scalars(self._rows)


class _FakeDB:
    """Minimal AsyncSession double: no cards/memories belong to the caller,
    but db.get() resolves one FOREIGN card (simulating an LLM-echoed ID)."""

    def __init__(self, foreign_card):
        self._foreign = foreign_card
        self.committed = False

    async def execute(self, stmt):
        return _Result(())

    async def get(self, model, pk):
        assert pk == self._foreign.id
        return self._foreign

    async def commit(self):
        self.committed = True


def _user(uid: uuid.UUID):
    return SimpleNamespace(id=uid)


@pytest.mark.asyncio
async def test_refresh_ignores_foreign_insight_ids(monkeypatch):
    me, other = uuid.uuid4(), uuid.uuid4()
    foreign = SimpleNamespace(
        id=uuid.uuid4(),
        user_id=other,
        status=InsightStatusEnum.NEW.value,
        relevance_score=0.5,
    )
    llm_updates = [
        {"insight_id": str(foreign.id), "action": "dismiss"},
        {"insight_id": str(foreign.id), "action": "boost", "new_insight_score": 0.99},
    ]
    monkeypatch.setattr(
        insights_router, "refresh_insights", AsyncMock(return_value=llm_updates)
    )
    monkeypatch.setattr(
        insights_router.CacheInvalidation,
        "invalidate_pattern",
        AsyncMock(return_value=None),
    )
    db = _FakeDB(foreign)

    response = await insights_router.refresh_insights_endpoint(_user(me), db, None)

    assert foreign.status == InsightStatusEnum.NEW.value
    assert foreign.relevance_score == 0.5
    assert response.updated_count == 0
    assert response.expired_count == 0


@pytest.mark.asyncio
async def test_refresh_applies_own_insight_ids(monkeypatch):
    """The ownership guard must not break the legitimate path."""
    me = uuid.uuid4()
    own = SimpleNamespace(
        id=uuid.uuid4(),
        user_id=me,
        title="t",
        summary="s",
        status=InsightStatusEnum.NEW.value,
        relevance_score=0.5,
        created_at=None,
    )
    db = _FakeDB(own)

    async def fake_execute(stmt):
        compiled = str(stmt)
        if "insight_cards" in compiled or "InsightCard" in type(stmt).__name__:
            return _Result((own,))
        return _Result(())

    db.execute = fake_execute  # type: ignore[method-assign]
    monkeypatch.setattr(
        insights_router,
        "refresh_insights",
        AsyncMock(
            return_value=[{"insight_id": str(own.id), "action": "dismiss"}]
        ),
    )
    monkeypatch.setattr(
        insights_router.CacheInvalidation,
        "invalidate_pattern",
        AsyncMock(return_value=None),
    )

    response = await insights_router.refresh_insights_endpoint(_user(me), db, None)

    assert own.status == InsightStatusEnum.DISMISSED.value
    assert response.updated_count == 1
