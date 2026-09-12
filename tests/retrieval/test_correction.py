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


def test_state_of_precedence():
    assert C.state_of(_mem({"cm_derived_dirty": True})) == "dirty"
    assert C.state_of(_mem({"cm_needs_check": True})) == "needs-check"
    assert C.state_of(_mem({"cm_superseded_by": "n1"})) == "superseded"
    # superseded outranks dirty; dirty outranks needs-check
    assert C.state_of(_mem({"cm_superseded_by": "n1", "cm_derived_dirty": True})) == "superseded"
    assert C.state_of(_mem({"cm_derived_dirty": True, "cm_needs_check": True})) == "dirty"


def test_slot_normalizes_once():
    s = C.Slot.of("  Proj-X  DB ", "DB", "")
    assert s is not None
    assert (s.subject, s.attribute, s.scope) == ("proj-x db", "db", "")
    assert s.key == ("proj-x db", "db", "default")
    assert s.has_scope is False
    assert C.Slot.of("", "db", "prod") is None
    assert C.Slot.of("p", "", "prod") is None
    # matches() normalizes the stored side, so legacy rows still hit
    assert s.matches(_mem({"cm_subject": "PROJ-X  DB", "cm_attribute": "DB"})) is True


def test_depends_on_predicate():
    m = _mem({"cm_derived_from": ["A", "B"]})
    assert C._depends_on(m, {"B"}) is True
    assert C._depends_on(m, {"Z"}) is False
    assert C._depends_on(_mem(None), {"A"}) is False


def test_normalize_slot():
    assert C.normalize_slot("  Proj-X  DB ") == "proj-x db"
    assert C.normalize_slot(None) == ""


def test_needs_rewrite_pure_rule():
    assert C.needs_rewrite("db prod la gi") is False
    # Bare "no" is negation, not a pronoun — must not trigger the LLM path.
    assert C.needs_rewrite("no sqlite on prod db") is False
    assert C.needs_rewrite("nó chạy ở cổng nào của dự án đó") is True


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


def test_decide_is_pure_no_store():
    old = _mem({"cm_subject": "proj-x", "cm_attribute": "db", "cm_scope": "prod"})
    status, meta, exact = C.decide_correction([old], slot=C.Slot.of("proj-x", "db", "prod"))
    assert (status, exact) == ("superseded", [old])
    assert C.CM_NEEDS_CHECK not in meta
    # bad valid_from poisons only the meta, still no store touched
    status2, meta2, _ = C.decide_correction([], valid_from="not-a-date")
    assert status2 == "added" and meta2[C.CM_NEEDS_CHECK] is True


async def test_resolve_supersede_chain_single_commit():
    from app.retrieval.memory.correction import Slot, resolve_correction
    uid = uuid.uuid4()
    old = _mem({"cm_subject": "proj-x", "cm_attribute": "db", "cm_scope": "prod"})
    old.user_id = uid
    db = _FakeDB(rows=[old])
    out = await resolve_correction(db, user_id=uid, title="DB", content="Postgres",
        slot=Slot.of("Proj-X", "db", "prod"))
    assert out["status"] == "superseded"
    assert db.committed == 1
    new = out["memory"]
    assert new.extra_metadata["cm_supersedes"] == str(old.id)
    assert old.extra_metadata["cm_superseded_by"] == str(new.id)
    assert out["superseded"] == [str(old.id)]


async def test_resolve_ambiguous_scope_keeps_both():
    from app.retrieval.memory.correction import Slot, resolve_correction
    uid = uuid.uuid4()
    old = _mem({"cm_subject": "proj-x", "cm_attribute": "db", "cm_scope": "prod"})
    old.user_id = uid
    db = _FakeDB(rows=[old])
    out = await resolve_correction(db, user_id=uid, title="DB", content="SQLite",
        slot=Slot.of("proj-x", "db", ""))
    assert out["status"] == "needs-check"
    assert old.extra_metadata.get("cm_superseded_by") is None
    assert out["memory"].extra_metadata["cm_needs_check"] is True


async def test_collect_derived_ids():
    from app.retrieval.memory.correction import collect_derived_ids
    uid = uuid.uuid4()
    target = _mem({})
    target.user_id = uid
    view = _mem({"cm_derived_from": [str(target.id)]})
    view.user_id = uid
    other = _mem({})
    other.user_id = uid
    db = _FakeDB(rows=[target, view, other])
    out = await collect_derived_ids(db, uid, [target.id])
    assert out == [view.id]
