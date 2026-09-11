"""Task 1 tests: cm_* metadata helpers + pure correction rules."""
from __future__ import annotations

import uuid

from app.models.memory import Memory
from app.retrieval.memory import correction as C


def _mem(meta: dict | None) -> Memory:
    return Memory(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        content="x",
        tags=[],
        extra_metadata=meta or {},
    )


def test_state_of_legacy_is_current():
    assert C.state_of(_mem(None)) == "current"


def test_normalize_slot():
    assert C.normalize_slot("  Proj-X  DB ") == "proj-x db"
    assert C.normalize_slot(None) == ""


def test_needs_rewrite_pure_rule():
    assert C.needs_rewrite("db prod la gi") is False
    assert C.needs_rewrite("no chay tren cong nao cua du an do") is True


def test_derived_dependents():
    old = _mem({"cm_derived_from": ["A", "B"]})
    assert C.find_derived_dependent_ids([old], {"B"}) == [str(old.id)]
    assert C.find_derived_dependent_ids([old], {"Z"}) == []
