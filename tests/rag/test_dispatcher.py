"""Tests for SourceSyncService idempotency + durable intents (dispatcher.py).

Regression coverage for three classes of bug:
1. Lazy-loading MemorySource.memory on an AsyncSession raises MissingGreenlet
   (the lookup eager-loads with selectinload).
2. Re-sync of previously-synced items must skip/update, not re-error.
3. Every created/updated row leaves exactly one durable index intent in the
   batch transaction, and a suppressed identity is skipped outright.

Real SQLite: the ``db`` fixture (tests/rag/conftest.py) builds a private temp
file, so these tests need no live Postgres (they used to error under
``--confcutdir=tests/rag`` for want of the root ``db`` fixture).
"""
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from app import database
from app.ingestion.dispatcher import SourceSyncService
from app.ingestion.document_memory import suppress_source_async
from app.ingestion.types import ConnectorItem, SyncResult
from app.models.index_outbox import IndexOutbox
from app.models.memory import Memory
from app.models.source import Source
from app.models.user import User

pytestmark = pytest.mark.asyncio


def _item(source_ref: str, content: str = "hello world") -> ConnectorItem:
    return ConnectorItem(
        title="Doc title",
        content=content,
        summary="summary",
        source_ref=source_ref,
        source_url="https://example.com/doc",
        tags=["test"],
    )


async def _outbox_rows() -> list[IndexOutbox]:
    """Read the outbox through a fresh session (immune to snapshot staleness)."""
    async with database.AsyncSessionLocal() as session:
        return list((await session.execute(select(IndexOutbox).order_by(IndexOutbox.seq))).scalars().all())


@pytest.fixture
async def user(db):
    u = User(email=f"sync-test-{uuid4()}@example.com", hashed_password=None)
    db.add(u)
    await db.flush()
    return u


@pytest.fixture
async def source(db, user):
    s = Source(
        user_id=user.id,
        source_type="manual",
        display_name=f"test-source-{uuid4()}",
        config={},
    )
    db.add(s)
    await db.flush()
    return s


async def test_resync_of_unchanged_item_skips(db, source):
    svc = SourceSyncService(db)
    item = _item("ref-1")

    outcome1, memory_id1 = await svc._persist_item(source, item)
    await db.flush()

    # Second persist of the same (source, ref) exercises the
    # MemorySource.memory relationship load — previously MissingGreenlet.
    outcome2, memory_id2 = await svc._persist_item(source, item)
    await db.flush()

    assert outcome1 == "added"
    assert outcome2 == "skipped"
    assert memory_id1 == memory_id2


async def test_resync_of_changed_item_updates(db, source):
    svc = SourceSyncService(db)
    await svc._persist_item(source, _item("ref-2", content="v1"))
    await db.flush()

    outcome, memory_id = await svc._persist_item(source, _item("ref-2", content="v2"))
    await db.flush()

    assert outcome == "updated"

    memory = await db.get(Memory, UUID(memory_id))
    assert memory.content == "v2"


async def test_batch_commit_leaves_one_intent_per_changed_row(db, source):
    svc = SourceSyncService(db)
    item1, item2 = _item("ref-1", content="v1"), _item("ref-2", content="v1")

    _, m1 = await svc._persist_item(source, item1)          # added
    _, m2 = await svc._persist_item(source, item2)          # added
    skipped, _ = await svc._persist_item(source, item1)     # unchanged -> no write
    updated, _ = await svc._persist_item(source, _item("ref-2", content="v2"))

    assert (skipped, updated) == ("skipped", "updated")

    # Same transaction as the batch: nothing is durable until _finalize commits.
    async with database.AsyncSessionLocal() as peek:
        assert (await peek.execute(select(IndexOutbox))).scalars().all() == []

    result = SyncResult(source_id=str(source.id), started_at=datetime.now(UTC),
                        finished_at=datetime.now(UTC))
    await svc._finalize(source, result)

    rows = await _outbox_rows()
    assert sorted((r.entity_id, r.revision, r.operation) for r in rows) == sorted([
        (UUID(m1).hex, 1, "upsert"),   # created
        (UUID(m2).hex, 1, "upsert"),   # created
        (UUID(m2).hex, 2, "upsert"),   # updated content -> revision 2
    ])
    assert {r.tenant_id for r in rows} == {source.user_id.hex}
    assert {r.status for r in rows} == {"pending"}


async def test_suppressed_item_is_skipped_and_not_recreated(db, source):
    svc = SourceSyncService(db)
    await suppress_source_async(db, user_id=source.user_id, source_ref="ref-sup")
    await db.commit()

    outcome, memory_id = await svc._persist_item(source, _item("ref-sup"))

    assert (outcome, memory_id) == ("skipped", None)   # counted as skipped
    assert (await db.execute(select(Memory))).scalars().all() == []
    assert (await db.execute(select(IndexOutbox))).scalars().all() == []


async def test_index_attempt_is_scoped_to_the_source_owner(db, source, monkeypatch):
    from app.retrieval.memory import write_back

    upserted: list[str] = []

    async def fake_upsert(memory):
        upserted.append(str(memory.id))
        return True

    monkeypatch.setattr(write_back, "safe_upsert_to_index", fake_upsert)

    def fake_enqueue(_memory_id):  # the helper is a plain function (R27(p2))
        return None

    monkeypatch.setattr(write_back, "safe_enqueue_graph_build", fake_enqueue)

    svc = SourceSyncService(db)
    stranger = User(email=f"stranger-{uuid4()}@example.com", hashed_password=None)
    db.add(stranger)
    await db.flush()
    foreign = Memory(id=uuid4(), user_id=stranger.id, content="foreign", tags=[])
    db.add(foreign)
    await db.commit()

    # A foreign id smuggled into the batch is never embedded on this source.
    await svc._index_memories([str(foreign.id)], user_id=source.user_id)
    assert upserted == []

    # The row's own owner still gets it embedded.
    await svc._index_memories([str(foreign.id)], user_id=stranger.id)
    assert upserted == [str(foreign.id)]
