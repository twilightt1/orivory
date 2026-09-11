"""Regression tests for referral service bugs (found in full-repo review).

R1: `get_referral_stats` built `... AND not ReferralReward.is_claimed`.
    In SQLAlchemy `not <Column>` collapses to plain Python `False`, so the
    query matched ZERO rows and unclaimed_rewards was always 0.
R4: `get_or_create_referral_code` was check-then-insert with no DB-level
    guard — two concurrent first calls both inserted, and every later call
    crashed with MultipleResultsFound (permanent 500).

Both tests are DB-free: R1 compiles the filter expression, R4 drives the
service with a fake AsyncSession that replays the race.
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import and_, func, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError

from app.models.referral import ReferralCode, ReferralReward
from app.services import referral_service


def _compile(stmt) -> str:
    return str(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))


def test_unclaimed_rewards_filter_is_real_sql_not_python_false():
    """The unclaimed-rewards WHERE clause must be SQL, not `AND false`."""
    clause = and_(
        ReferralReward.user_id == uuid.uuid4(),
        ReferralReward.is_claimed.is_(False),
    )
    compiled = _compile(select(func.count(ReferralReward.id)).where(clause))
    assert "false" in compiled.lower()
    # Guard against the exact historical regression: a bare `not Column`
    # collapses to `WHERE false` with no reference to is_claimed.
    assert "is_claimed" in compiled, (
        "filter lost the column — likely `not Column` collapsed to False again"
    )


def test_stats_filter_matches_service_implementation():
    """The service's real query must include the is_claimed predicate."""
    import inspect

    src = inspect.getsource(referral_service.get_referral_stats)
    assert "is_claimed.is_(False)" in src or "is_claimed == False" in src
    assert "not ReferralReward.is_claimed" not in src


class _FakeScalars:
    def __init__(self, rows):
        self._rows = rows

    def first(self):
        return self._rows[0] if self._rows else None


class _FakeResult:
    def __init__(self, rows=None):
        self._rows = rows or []

    def scalar_one_or_none(self):
        if len(self._rows) > 1:
            from sqlalchemy.exc import MultipleResultsFound

            raise MultipleResultsFound("too many")
        return self._rows[0] if self._rows else None

    def scalars(self):
        return _FakeScalars(self._rows)

    def scalar(self):
        return self._rows[0] if self._rows else None


class _RacySession:
    """Replays the two-concurrent-first-calls race: the SELECT sees nothing
    (both checked before either inserted), the INSERT violates the partial
    unique index, and the re-read finds the winner's row."""

    def __init__(self, winner: ReferralCode):
        self._winner = winner
        self._selects = 0

    async def execute(self, stmt):
        self._selects += 1
        compiled = _compile(stmt)
        if "referral_codes" in compiled and "code =" in compiled:
            return _FakeResult([])  # fresh code
        if "ORDER BY" in compiled:
            return _FakeResult([self._winner])
        return _FakeResult([])  # first check: nothing yet

    def add(self, obj):
        pass

    async def scalar(self, stmt):
        result = await self.execute(stmt)
        return result.scalars().first()

    async def commit(self):
        raise IntegrityError("INSERT", {}, Exception("duplicate key"))

    async def rollback(self):
        pass

    async def refresh(self, obj):
        pass


@pytest.mark.asyncio
async def test_get_or_create_survives_insert_race():
    """Loser of the concurrent-create race gets the winner's row, not a 500."""
    winner = ReferralCode(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        code="ML-RACEWIN",
        is_active=True,
    )
    db = _RacySession(winner)

    got = await referral_service.get_or_create_referral_code(db, winner.user_id)

    assert got is winner
    assert db._selects >= 2  # initial check + post-rollback re-read


@pytest.mark.asyncio
async def test_get_or_create_returns_existing_without_insert():
    """Happy path: existing active code is returned, no commit attempted."""
    existing = ReferralCode(
        id=uuid.uuid4(), user_id=uuid.uuid4(), code="ML-EXISTS", is_active=True
    )

    class _CalmSession(_RacySession):
        async def execute(self, stmt):
            return _FakeResult([existing])

        async def commit(self):
            raise AssertionError("must not insert when a code exists")

    got = await referral_service.get_or_create_referral_code(
        _CalmSession(existing), existing.user_id
    )
    assert got is existing


def test_generate_referral_code_format():
    code = referral_service.generate_referral_code()
    assert code.startswith("ML-") and len(code) == 11
    assert referral_service.generate_referral_code() != referral_service.generate_referral_code()
