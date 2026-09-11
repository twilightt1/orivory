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


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows

    def first(self):
        return self._rows[0] if self._rows else None


class _FakeDB:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.added = []
        self.deleted = []
        self.committed = 0

    async def get(self, model, obj_id):
        for row in self.rows:
            if getattr(row, "id", None) == obj_id:
                return row
        return None

    async def execute(self, _stmt):
        return _FakeResult(self.rows)

    def add(self, obj):
        self.added.append(obj)

    async def delete(self, obj):
        self.deleted.append(obj)

    async def commit(self):
        self.committed += 1

    async def refresh(self, _obj):
        return None


async def test_resolve_supersede_chain_single_commit():
    from app.retrieval.memory.correction import resolve_correction
    uid = uuid.uuid4()
    old = _mem({"cm_subject": "proj-x", "cm_attribute": "db", "cm_scope": "prod"})
    old.user_id = uid
    db = _FakeDB(rows=[old])
    out = await resolve_correction(db, user_id=uid, title="DB", content="Postgres",
        subject="Proj-X", attribute="db", scope="prod")
    assert out["status"] == "superseded"
    assert db.committed == 1
    new = out["memory"]
    assert new.extra_metadata["cm_supersedes"] == str(old.id)
    assert old.extra_metadata["cm_superseded_by"] == str(new.id)
    assert out["superseded"] == [str(old.id)]


async def test_resolve_ambiguous_scope_keeps_both():
    from app.retrieval.memory.correction import resolve_correction
    uid = uuid.uuid4()
    old = _mem({"cm_subject": "proj-x", "cm_attribute": "db", "cm_scope": "prod"})
    old.user_id = uid
    db = _FakeDB(rows=[old])
    out = await resolve_correction(db, user_id=uid, title="DB", content="SQLite",
        subject="proj-x", attribute="db", scope="")
    assert out["status"] == "needs-check"
    assert old.extra_metadata.get("cm_superseded_by") is None
    assert out["memory"].extra_metadata["cm_needs_check"] is True
