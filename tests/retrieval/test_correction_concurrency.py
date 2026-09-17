"""P4b T2 — same-slot revision CAS + late-import chronology.

Real SQLite (``tests.retrieval.test_visibility``'s private per-test file): the
CAS answer has to come from the row's revision in the DB, never from what one
session happens to hold in memory. A "second writer" is either a second
connection's committed write or a hook-driven interleave — nothing here sleeps.

The hazard these pin (spec §8.2): two writers on one slot must not end up with
two current facts or with one silently superseding on a snapshot the other one
already moved past.
"""
from __future__ import annotations

from sqlalchemy import select, update

from app import database
from app.models.memory import Memory
from app.retrieval.memory import correction as C
from app.retrieval.memory.correction import Slot, resolve_correction
from tests.retrieval.test_visibility import _mem, _owner, db  # noqa: F401

SLOT = Slot.of("proj-x", "db", "prod")
SLOT_META = {"cm_subject": "proj-x", "cm_attribute": "db", "cm_scope": "prod"}


def _slotted(owner, title: str, *, valid_from: str | None = None) -> Memory:
    meta = dict(SLOT_META)
    if valid_from is not None:
        meta[C.CM_VALID_FROM] = valid_from
    return _mem(owner, title, meta=meta)


async def _rows(db, owner) -> dict:
    """Owner's rows keyed by id, read FRESH (never from the identity map)."""
    db.expire_all()
    return {row.id: row for row in (await db.execute(
        select(Memory).where(Memory.user_id == owner))).scalars().all()}


# ── same-slot CAS ───────────────────────────────────────────────────────────


async def test_same_slot_second_writer_with_a_stale_snapshot_conflicts(db):
    """Two writers on one slot: the first supersedes, the second decided on the
    snapshot the first moved past → ``conflict``, its whole row flagged for a
    check, and exactly ONE current fact left on the slot."""
    owner = await _owner(db)
    first = _slotted(owner, "db v1")
    db.add(first)
    await db.commit()
    snapshot = {first.id: first.revision}  # what BOTH writers read

    winner = await resolve_correction(db, user_id=owner, title="DB", content="Postgres",
                                      slot=SLOT, expected_revisions=dict(snapshot))
    assert winner["status"] == "superseded"
    winner_id, first_id = winner["memory"].id, first.id

    loser = await resolve_correction(db, user_id=owner, title="DB", content="SQLite",
                                     slot=SLOT, expected_revisions=dict(snapshot))
    assert loser["status"] == "conflict", "a stale snapshot never supersedes silently"
    assert (loser["superseded"], loser["dirtied"]) == ([], [])
    assert loser["memory"].extra_metadata[C.CM_NEEDS_CHECK] is True
    assert C.CM_SUPERSEDES not in loser["memory"].extra_metadata
    loser_id = loser["memory"].id

    rows = await _rows(db, owner)
    assert C.state_of(rows[first_id]) == "superseded"
    assert rows[first_id].extra_metadata[C.CM_SUPERSEDED_BY] == str(winner_id)
    assert [mid for mid, row in rows.items() if C.state_of(row) == "current"] == [winner_id]
    assert C.state_of(rows[loser_id]) == "needs-check"


async def test_a_candidate_written_by_another_writer_conflicts(db):
    """The named candidate's revision moved (any other write bumps it, §2): the
    caller's expected revision is void and nothing is superseded."""
    owner = await _owner(db)
    row = _slotted(owner, "db v1")
    db.add(row)
    await db.commit()
    snapshot = {row.id: row.revision}
    row_id = row.id

    async with database.AsyncSessionLocal() as other:  # a second connection's commit
        await other.execute(update(Memory).where(Memory.id == row_id)
                            .values(revision=Memory.revision + 1))
        await other.commit()

    out = await resolve_correction(db, user_id=owner, title="DB", content="SQLite",
                                   slot=SLOT, expected_revisions=snapshot)

    assert out["status"] == "conflict"
    assert out["superseded"] == []
    assert out["memory"].extra_metadata[C.CM_NEEDS_CHECK] is True
    rows = await _rows(db, owner)
    assert rows[row_id].extra_metadata.get(C.CM_SUPERSEDED_BY) is None
    assert rows[row_id].revision == snapshot[row_id] + 1  # the other writer's bump stands


async def test_the_apply_is_guarded_when_a_candidate_moves_mid_call(db, monkeypatch):
    """The apply itself is guarded: a candidate that moves after the candidate
    read is refused by the UPDATE's ``WHERE revision`` (rowcount 0), so the
    transaction stands down instead of writing over the newer state."""
    owner = await _owner(db)
    row = _slotted(owner, "db v1")
    db.add(row)
    await db.commit()
    row_id, expected_rev = row.id, row.revision

    real = C._cas_supersede
    raced = False

    async def _racing(db_, memory, successor_id, expected):
        nonlocal raced
        if not raced:  # the caller's snapshot was read; now another writer lands
            raced = True
            await db_.execute(update(Memory).where(Memory.id == memory.id)
                              .values(revision=Memory.revision + 1))
        return await real(db_, memory, successor_id, expected)

    monkeypatch.setattr(C, "_cas_supersede", _racing)

    out = await resolve_correction(db, user_id=owner, title="DB", content="SQLite",
                                   slot=SLOT, expected_revisions={row_id: expected_rev})

    assert raced, "the interleave really happened"
    assert out["status"] == "conflict"
    assert out["superseded"] == []
    assert out["memory"].extra_metadata[C.CM_NEEDS_CHECK] is True


async def test_an_unnamed_candidate_is_never_superseded(db):
    """A caller that vetted ONE row must not silently move a second one the slot
    gained meanwhile (e.g. the winner of the race above)."""
    owner = await _owner(db)
    first = _slotted(owner, "db v1")
    db.add(first)
    await db.commit()
    first_id, first_rev = first.id, first.revision

    second = _slotted(owner, "db v2")
    db.add(second)
    await db.commit()
    second_id = second.id

    out = await resolve_correction(db, user_id=owner, title="DB", content="SQLite",
                                   slot=SLOT, expected_revisions={first_id: first_rev})

    assert out["status"] == "conflict"
    assert out["superseded"] == []
    rows = await _rows(db, owner)
    assert rows[second_id].extra_metadata.get(C.CM_SUPERSEDED_BY) is None
    assert C.state_of(rows[first_id]) == "current"


# ── late-import chronology ──────────────────────────────────────────────────


async def test_late_old_import_needs_check_instead_of_superseding(db):
    """An older record imported after a newer fact: needs-check, never a
    supersede by ingestion order (spec §8.2)."""
    owner = await _owner(db)
    newer = _slotted(owner, "db v2", valid_from="2026-09-10T00:00:00+00:00")
    db.add(newer)
    await db.commit()
    newer_id = newer.id

    out = await resolve_correction(db, user_id=owner, title="DB", content="old record",
                                   slot=SLOT, valid_from="2026-09-01T00:00:00+00:00")

    assert out["status"] == "needs-check"
    assert out["superseded"] == []
    assert out["memory"].extra_metadata[C.CM_NEEDS_CHECK] is True
    rows = await _rows(db, owner)
    assert rows[newer_id].extra_metadata.get(C.CM_SUPERSEDED_BY) is None
    assert C.state_of(rows[newer_id]) == "current"


async def test_a_newer_valid_from_still_supersedes(db):
    """The chronology rule is a comparison, not a ban: newer event time wins."""
    owner = await _owner(db)
    older = _slotted(owner, "db v1", valid_from="2026-09-01T00:00:00+00:00")
    db.add(older)
    await db.commit()
    older_id = older.id

    out = await resolve_correction(db, user_id=owner, title="DB", content="newer record",
                                   slot=SLOT, valid_from="2026-09-10T00:00:00+00:00")

    assert out["status"] == "superseded"
    assert out["superseded"] == [str(older_id)]


async def test_an_unparseable_candidate_stamp_does_not_block_a_supersede(db):
    """A junk stamp on the stored row is not a chronology claim: it must not
    turn every later correction into needs-check (legacy rows carry anything)."""
    owner = await _owner(db)
    row = _slotted(owner, "db v1", valid_from="not-a-date")
    db.add(row)
    await db.commit()
    row_id = row.id

    out = await resolve_correction(db, user_id=owner, title="DB", content="SQLite",
                                   slot=SLOT, valid_from="2026-09-10T00:00:00+00:00")

    assert out["status"] == "superseded"
    assert out["superseded"] == [str(row_id)]
