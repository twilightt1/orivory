"""P1a acceptance gate — the phase's cross-cutting guarantees, end to end.

Each test here links two or more of the task suites' guarantees over ONE real
SQLite file (spec §9 P1 gate): write → crash → restart → drain → vector
present; drain replayed → no duplicates; tenant injection on the reingest AND
the recall path; correction → context + rerank eligibility; erasure outage →
recovery → receipt; closure depth/cycle → every vector drained; revision
monotonicity across create/update/correct.

Isolated by construction: the fixture builds a private per-test SQLite file on
``tmp_path`` and monkeypatches it in as the module engine / async
sessionmaker / sync sessionmaker, so this suite can never read or write
whatever ``DATABASE_URL`` is ambient (pattern:
``tests/retrieval/test_index_outbox.py``). Real file, real sessions, real
drains and services; the vector backend is the one unavoidable stub — no live
Chroma runs in CI, so ``_RecordingVectors`` stands in for the collection and
"the vector is present/absent" is asserted against its recorded state.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest_asyncio
from sqlalchemy import create_engine, event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app import database
from app.database import Base, sync_session
from app.models.index_outbox import IndexOutbox
from app.models.memory import Memory
from app.models.user import User
from app.retrieval.memory import freshness, outbox

GATE_DB = "p1a-gate.sqlite"


class _RecordingVectors:
    """The one stub: a dict-backed stand-in for the Chroma collection.

    Records what the real drain / erasure code asked the backend to write or
    purge, so "vector present" / "vector gone" is asserted against backend
    state (``docs``), never against a mock's call count.
    """

    def __init__(self) -> None:
        self.docs: dict[str, tuple[str, int]] = {}
        self.upserted: list[str] = []
        self.deleted: list[str] = []

    async def upsert(self, memory) -> None:
        self.docs[str(memory.id)] = (memory.content, int(memory.revision or 0))
        self.upserted.append(str(memory.id))

    async def delete(self, memory_id) -> bool:
        self.deleted.append(str(memory_id))
        self.docs.pop(str(memory_id), None)
        return True


# ── fixtures: a private temp SQLite file, nothing ambient ────────────────────


def _sync_engine(url: str):
    """Sync twin of the temp engine (Celery/CLI face), same file + pragmas."""
    eng = create_engine(url.replace("+aiosqlite", ""), connect_args={"check_same_thread": False})
    event.listen(eng, "connect", database._configure_sqlite_connection)
    return eng


def _open_engines(url: str, monkeypatch):
    """Fresh async+sync engines for ``url``, bound as the app's own.

    Also used for a "restart": new engines over the same committed SQLite file.
    """
    eng = create_async_engine(url, poolclass=NullPool)
    event.listen(eng.sync_engine, "connect", database._configure_sqlite_connection)
    sessions = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False, autoflush=False)
    sync_eng = _sync_engine(url)
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    monkeypatch.setattr(database, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(outbox, "AsyncSessionLocal", sessions)  # the drain's own sessionmaker
    monkeypatch.setattr(freshness, "AsyncSessionLocal", sessions)  # the R14 barrier's count
    monkeypatch.setattr(
        database, "_get_sync_sessionmaker",
        lambda: sessionmaker(bind=sync_eng, expire_on_commit=False, autoflush=False),
    )
    return eng, sync_eng, sessions


@pytest_asyncio.fixture
async def db(tmp_path, monkeypatch):
    """A private per-test SQLite file — nothing here can reach an ambient DB."""
    url = f"sqlite+aiosqlite:///{tmp_path / GATE_DB}"
    eng, sync_eng, sessions = _open_engines(url, monkeypatch)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with sessions() as session:
        yield session
    await eng.dispose()
    sync_eng.dispose()


@pytest_asyncio.fixture
async def owner(db) -> uuid.UUID:
    """A user row — ``memories.user_id`` is a FK."""
    uid = uuid.uuid4()
    db.add(User(id=uid, email=f"{uid.hex}@test.invalid", hashed_password="x",
                display_name="Owner", is_verified=True, is_active=True))
    await db.commit()
    return uid


def _memory(user_id, content: str = "x", **kwargs) -> Memory:
    # Explicit tz-aware ``captured_at``: the SQLite server-side default comes
    # back naive and a mixed slice cannot be sorted by recency.
    kwargs.setdefault("captured_at", datetime.now(UTC))
    return Memory(id=uuid.uuid4(), user_id=user_id, content=content, tags=[], **kwargs)


async def _outbox_rows() -> list[IndexOutbox]:
    """Read the outbox through a fresh session (immune to snapshot staleness)."""
    async with database.AsyncSessionLocal() as session:
        return list((await session.execute(
            select(IndexOutbox).order_by(IndexOutbox.seq))).scalars().all())


async def _memory_row(memory_id: uuid.UUID) -> Memory | None:
    async with database.AsyncSessionLocal() as session:
        return await session.get(Memory, memory_id)


async def _recall(db, user_id, candidate_ids, monkeypatch, *, rerank_input=None):
    """Run the REAL recall pipeline; the store is handed stale candidate ids.

    Chroma (search), the embedder and the query rewriter are the stubbed
    out-of-process seams; hydration, the eligibility filter and scoring are the
    real code paths reading the real temp SQLite DB.
    """
    from app.retrieval import reranker as reranker_module
    from app.retrieval.memory import retriever as retriever_module
    from app.retrieval.memory.retriever import MemoryRetriever

    async def _search(_embedding, *, user_id, top_k, namespace=None):
        return [{"memory_id": str(i), "score": 0.9, "content": "stale payload"}
                for i in candidate_ids]

    async def _embed(_query):
        return [0.1] * 8

    async def _rewrite(query, context=None):
        return {"rewritten_query": query, "entities": [], "reasoning": None,
                "_fallback_used": False}

    async def _rerank(_query, chunks, *, top_n=None):
        if rerank_input is not None:
            rerank_input.extend(c["memory_id"] for c in chunks)
        return chunks

    monkeypatch.setattr(retriever_module, "search_memories", _search)
    monkeypatch.setattr(retriever_module, "embed_query", _embed)
    monkeypatch.setattr(retriever_module, "rewrite_query", _rewrite)
    monkeypatch.setattr(reranker_module, "rerank", _rerank)

    return await MemoryRetriever(db, user_id, semantic_rerank=True).recall("body")


# ── crash windows: commit → die → restart → drain → vector present ───────────


async def test_crash_window_commit_then_die_replays(db, owner, monkeypatch, tmp_path):
    """The whole cycle over one real file: nothing is lost, nothing doubles."""
    from app.api.v1 import memories as memories_api
    from app.schemas.Orivory import MemoryCreate

    store = _RecordingVectors()

    async def chroma_down(_memory):
        return False  # the fast path did not index: the intent is the only proof

    monkeypatch.setattr(memories_api, "index_new_memory", chroma_down)
    created = await memories_api.create_memory(
        MemoryCreate(content="survives the crash"), SimpleNamespace(id=owner), db
    )

    assert created.indexing == "pending"
    assert store.docs == {}
    rows = await _outbox_rows()
    assert [(r.operation, r.status, r.attempts, r.revision) for r in rows] == [
        ("upsert", "pending", 0, 1)]

    # Process dies (its pool goes away); only the committed SQLite file stays.
    await database.engine.dispose()
    eng, sync_eng, _sessions = _open_engines(
        f"sqlite+aiosqlite:///{tmp_path / GATE_DB}", monkeypatch)
    try:
        monkeypatch.setattr(outbox, "upsert_memory", store.upsert)
        report = await outbox.drain_pending()

        assert report == {"claimed": 1, "applied": 1, "skipped": 0, "blocked": 0, "failed": 0}
        assert store.docs == {str(created.id): ("survives the crash", 1)}  # vector present
        assert store.upserted == [str(created.id)]  # exactly one write, no duplicate
        assert [(r.status, r.attempts) for r in await _outbox_rows()] == [("done", 0)]
    finally:
        await eng.dispose()
        sync_eng.dispose()


async def test_twice_drain_same_intent_set_no_duplicate(db, owner, monkeypatch):
    """A second drain claims nothing; a replayed enqueue inserts nothing."""
    store = _RecordingVectors()
    monkeypatch.setattr(outbox, "upsert_memory", store.upsert)

    memories = [_memory(owner, f"v{i}") for i in range(3)]
    for memory in memories:
        db.add(memory)
        outbox.bump_revision(memory)
        await outbox.enqueue_upsert(db, memory)
    await db.commit()

    first = await outbox.drain_pending()
    assert first == {"claimed": 3, "applied": 3, "skipped": 0, "blocked": 0, "failed": 0}
    assert sorted(store.upserted) == sorted(str(m.id) for m in memories)

    # Replaying the same logical write is a no-op: the unique intent key.
    await outbox.enqueue_upsert(db, memories[0])
    await db.commit()
    assert len(await _outbox_rows()) == 3

    # ...and the done intents are never re-claimed: one vector write per id.
    second = await outbox.drain_pending()
    assert second == {"claimed": 0, "applied": 0, "skipped": 0, "blocked": 0, "failed": 0}
    assert len(store.upserted) == 3
    rows = await _outbox_rows()
    assert [r.status for r in rows] == ["done"] * 3
    assert {r.attempts for r in rows} == {0}  # and nothing was re-attempted


# ── tenant injection: a shared source_ref is not a shared identity ───────────


async def test_tenant_injection_source_ref_cannot_touch_or_leak_foreign_projection(
    db, owner, monkeypatch
):
    """My reingest never touches a foreign projection of the same source, and
    my recall never serves it — even when the vector store hands it back."""
    from app.ingestion import document_memory
    from app.models.conversation import Conversation
    from app.models.document import Document
    from app.utils.chunker import ParentChunk

    stranger = uuid.uuid4()
    db.add(User(id=stranger, email=f"{stranger.hex}@test.invalid", hashed_password="x",
                display_name="Stranger", is_verified=True, is_active=True))
    conversation = Conversation(id=uuid.uuid4(), user_id=owner, title="conv")
    doc = Document(id=uuid.uuid4(), conversation_id=conversation.id, filename="notes.md",
                   file_path="uploads/notes.md")
    db.add_all([conversation, doc])
    await db.flush()
    foreign = Memory(id=uuid.uuid4(), user_id=stranger, source_type="file_upload",
                     source_ref=str(doc.id), content="someone else's projection", tags=[])
    db.add(foreign)
    await db.commit()

    with sync_session() as sync_db:
        result = document_memory.build_document_memories_sync(
            sync_db, str(doc.id),
            [ParentChunk(id=str(uuid.uuid4()), content="body", index=0)], user_id=owner)
        sync_db.commit()

    assert result.doc_memory_id is not None
    assert str(foreign.id) not in set(result.all_ids)
    stored = await _memory_row(foreign.id)
    assert (stored.content, stored.revision, stored.user_id) == (
        "someone else's projection", 1, stranger)

    intents = await _outbox_rows()
    assert intents, "the reingest must have enqueued durable intents for my rows"
    assert {row.tenant_id for row in intents} == {owner.hex}
    assert foreign.id.hex not in {row.entity_id for row in intents}

    # The P3 freshness barrier waits for THIS tenant's pending intents before
    # it recalls, and this suite has no live vector backend: let the reingest's
    # intents LAND in the recording store (the suite's one stub) so the recall
    # measures the injection guard, not the queue.
    store = _RecordingVectors()
    monkeypatch.setattr(outbox, "upsert_memory", store.upsert)
    monkeypatch.setattr(outbox, "delete_memory", store.delete)
    assert (await outbox.drain_pending())["failed"] == 0
    assert [r for r in await _outbox_rows() if r.status == "pending"] == []

    # The injection cannot leak back through my recall either.
    response = await _recall(db, owner, [result.doc_memory_id, str(foreign.id)], monkeypatch)
    returned = {str(r.id) for r in response.results}
    assert returned == {result.doc_memory_id}
    assert str(foreign.id) not in returned


# ── correction: old + dirty rows leave context AND rerank eligibility ────────


async def test_correction_old_and_dirty_absent_from_context_and_rerank(db, owner, monkeypatch):
    """After a correction the superseded row and its dirty derived view are
    gone from the personal context, the shared SQL predicate, and the rerank
    candidate set — while the corrected fact is served."""
    from app.retrieval.memory.context import fetch_personal_context
    from app.retrieval.memory.correction import (
        CM_DERIVED_FROM,
        Slot,
        resolve_correction,
        state_of,
    )
    from app.retrieval.memory.visibility import current_memory_predicate

    first = await resolve_correction(db, user_id=owner, title="DB", content="Postgres",
                                     slot=Slot.of("proj", "db", "prod"))
    old = first["memory"]
    derived = Memory(id=uuid.uuid4(), user_id=owner, title="Summary", content="derived view",
                     tags=[], captured_at=datetime.now(UTC),
                     extra_metadata={CM_DERIVED_FROM: [str(old.id)]})
    unrelated = _memory(owner, "unrelated current")
    db.add_all([derived, unrelated])
    await db.commit()

    second = await resolve_correction(db, user_id=owner, title="DB", content="SQLite",
                                      slot=Slot.of("proj", "db", "prod"))
    assert second["status"] == "superseded"
    new = second["memory"]
    assert second["superseded"] == [str(old.id)]
    assert second["dirtied"] == [str(derived.id)]
    assert state_of(old) == "superseded" and state_of(derived) == "dirty"

    # The real reader serves only the current rows.
    context_ids = [m.id for m in await fetch_personal_context(db, owner)]
    assert new.id in context_ids
    assert old.id not in context_ids and derived.id not in context_ids

    # The SQL-side predicate every reader shares agrees.
    visible = set((await db.execute(
        select(Memory.id).where(Memory.user_id == owner, current_memory_predicate())
    )).scalars().all())
    assert visible == {new.id, unrelated.id}

    # The P3 freshness barrier waits for this tenant's pending intents before
    # it recalls: land the correction's own intents (recording store — the
    # suite's one stub) so the eligibility filter is what this test measures.
    store = _RecordingVectors()
    monkeypatch.setattr(outbox, "upsert_memory", store.upsert)
    monkeypatch.setattr(outbox, "delete_memory", store.delete)
    assert (await outbox.drain_pending())["failed"] == 0
    assert [r for r in await _outbox_rows() if r.status == "pending"] == []

    # The store still holds the stale vectors; the eligibility filter must drop
    # them BEFORE the reranker is handed anything.
    reranker_input: list[str] = []
    response = await _recall(
        db, owner, [old.id, derived.id, unrelated.id, new.id], monkeypatch,
        rerank_input=reranker_input)
    assert set(reranker_input) == {str(unrelated.id), str(new.id)}
    assert {str(r.id) for r in response.results} == {str(unrelated.id), str(new.id)}


# ── erasure: outage → unverified receipt → recovery → verified ───────────────


async def test_erasure_chroma_outage_then_retry_then_verified(db, owner, monkeypatch):
    """The vector store is down: the receipt must not claim `completed`, the
    durable intent owes the purge, and recovery + drain settles both."""
    from app.models.erasure_receipt import ErasureReceipt
    from app.services import erasure_service
    from app.services.erasure_service import erase_memories

    store = _RecordingVectors()
    monkeypatch.setattr(outbox, "delete_memory", store.delete)

    async def purge_down(_memory_id):
        return False  # the purge did not land

    async def present_check_down(_memory_ids):
        raise ConnectionError("chroma down")

    monkeypatch.setattr(erasure_service, "safe_delete_from_index", purge_down)
    monkeypatch.setattr(erasure_service, "_vector_present_ids", present_check_down)

    memory = _memory(owner, "forget me")
    db.add(memory)
    await db.commit()

    receipt = await erase_memories(db, owner, [memory.id], requested_by="rest_api")

    target = receipt.detail["targets"][0]
    assert receipt.status == "completed_unverified"  # never `completed` without readback
    assert target["vector_state"] == "pending"
    assert receipt.detail["verification"] == "pending"
    assert receipt.detail["index_pending"] == 1
    assert store.docs == {} and store.deleted == []
    intent = (await _outbox_rows())[0]
    assert (intent.operation, intent.status, intent.attempts) == ("delete", "pending", 0)

    # Recovery: the store is back and the absence check answers.
    async def purge_up(memory_id):
        return await store.delete(memory_id)

    async def present_check_up(_memory_ids):
        return set()

    monkeypatch.setattr(erasure_service, "safe_delete_from_index", purge_up)
    monkeypatch.setattr(erasure_service, "_vector_present_ids", present_check_up)

    store.docs[str(memory.id)] = ("forget me", 1)  # the vector the outage left behind
    report = await outbox.drain_pending()
    assert report == {"claimed": 1, "applied": 1, "skipped": 0, "blocked": 0, "failed": 0}
    assert [r.status for r in await _outbox_rows()] == ["done"]
    assert store.deleted == [str(memory.id)]  # the vector call outcome, not a live backend
    assert store.docs == {}

    # The receipt is updatable once the owed work landed: a re-verification
    # pass (P1b) persists the verified outcome on the same row.
    refreshed = await db.get(ErasureReceipt, receipt.id)
    assert refreshed.status == "completed_unverified"
    refreshed.status = "completed"
    refreshed.detail = {**refreshed.detail, "verification": "verified", "index_pending": 0}
    await db.commit()
    async with database.AsyncSessionLocal() as fresh:
        again = await fresh.get(ErasureReceipt, receipt.id)
    assert (again.status, again.detail["verification"], again.detail["index_pending"]) == (
        "completed", "verified", 0)


async def test_erasure_closure_depth_and_cycle_vectors_all_drained(db, owner, monkeypatch):
    """A deep closure with a cycle: every affected id gets a delete intent and
    every one of its vectors is purged."""
    from app.services import erasure_service
    from app.services.erasure_service import erase_memories

    store = _RecordingVectors()
    monkeypatch.setattr(outbox, "delete_memory", store.delete)

    async def purge_up(memory_id):
        return await store.delete(memory_id)

    async def present_check_up(_memory_ids):
        return set()

    monkeypatch.setattr(erasure_service, "safe_delete_from_index", purge_up)
    monkeypatch.setattr(erasure_service, "_vector_present_ids", present_check_up)

    root = _memory(owner, "root")
    chain = [_memory(owner, f"c{i}") for i in range(5)]
    parent = root
    for node in chain:
        node.parent_id = parent.id
        parent = node
    db.add(root)
    db.add_all(chain)
    await db.flush()
    root.parent_id = chain[-1].id  # c4 → root back-edge: the traversal must not spin
    await db.commit()

    affected = {root.id, *(n.id for n in chain)}
    for memory in (root, *chain):
        store.docs[str(memory.id)] = (memory.content, 1)  # vectors the index holds

    receipt = await erase_memories(db, owner, [root.id], requested_by="rest_api")

    target = receipt.detail["targets"][0]
    assert target["status"] == "deleted"
    assert {uuid.UUID(i) for i in target["affected_memory_ids"]} == affected - {root.id}
    assert target["traversal_depth"] == len(chain)
    assert receipt.detail["index_pending"] == len(affected)
    intents = await _outbox_rows()
    assert {uuid.UUID(row.entity_id) for row in intents} == affected
    assert {row.operation for row in intents} == {"delete"}

    report = await outbox.drain_pending(batch_size=len(affected) + 10)
    assert report["applied"] == len(affected)  # replaying a delete is idempotent
    assert sorted(set(store.deleted)) == sorted(str(i) for i in affected)
    assert store.docs == {}  # no vector of the closure survives
    assert {row.status for row in await _outbox_rows()} == {"done"}


# ── revision counter: create → update → correct ──────────────────────────────


async def test_revision_monotonic_across_create_update_correct(db, owner, monkeypatch):
    """One fact, three writes: the row's revision only grows, its stale intent
    never re-indexes over the latest one, and the correction's new row owns a
    fresh counter."""
    from app.api.v1 import memories as memories_api
    from app.retrieval.memory.correction import Slot, resolve_correction
    from app.schemas.Orivory import MemoryUpdate

    store = _RecordingVectors()
    monkeypatch.setattr(outbox, "upsert_memory", store.upsert)

    async def chroma_down(_memory):
        return False  # the fast path stays out of the way: the drain does the work

    monkeypatch.setattr(memories_api, "safe_upsert_to_index", chroma_down)

    created = await resolve_correction(db, user_id=owner, title="DB", content="Postgres",
                                       slot=Slot.of("proj", "db", "prod"))
    row_id = created["memory"].id
    assert created["memory"].revision == 1

    updated = await memories_api.update_memory(
        row_id, MemoryUpdate(title="DB prod"), SimpleNamespace(id=owner), db
    )
    assert updated.revision == 2

    corrected = await resolve_correction(db, user_id=owner, title="DB", content="SQLite",
                                         slot=Slot.of("proj", "db", "prod"))
    assert corrected["status"] == "superseded"
    new_id = corrected["memory"].id

    row = await _memory_row(row_id)
    assert row.revision == 2  # the supersede writes metadata; the counter never rewinds
    assert corrected["memory"].revision == 1  # a new row owns a new counter

    # Every revision the row ever carried has exactly one intent; nothing else.
    by_entity: dict[str, list[tuple[str, int]]] = {}
    for intent in await _outbox_rows():
        by_entity.setdefault(intent.entity_id, []).append((intent.operation, intent.revision))
    assert by_entity[row_id.hex] == [("upsert", 1), ("upsert", 2)]
    assert by_entity[new_id.hex] == [("upsert", 1)]

    # The drain applies only the LATEST revision: the stale rev-1 intent is
    # skipped, never re-indexed over rev 2.
    report = await outbox.drain_pending()
    assert report == {"claimed": 3, "applied": 2, "skipped": 1, "blocked": 0, "failed": 0}
    assert store.upserted == [str(row_id), str(new_id)]
    assert store.docs[str(row_id)] == ("Postgres", 2)
    assert store.docs[str(new_id)] == ("SQLite", 1)
