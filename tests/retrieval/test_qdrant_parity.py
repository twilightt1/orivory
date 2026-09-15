"""T2 — memory generation on Qdrant: payload contract, filters, contract guard.

Isolated by construction: a real embedded Qdrant on a private ``tmp_path``
folder (closed in teardown, so its folder lock never leaks into another test),
or — when the environment asks for it (``QDRANT_MODE=server``, see the ``env``
fixture) — the live server, with the same private SQLite manifest and a private
per-test SQLite file monkeypatched in as the module engines /
sessionmakers (the ``tests/retrieval/test_index_outbox.py`` pattern), so this
suite can never read or write whatever ``DATABASE_URL`` is ambient.

Embeddings are deterministic unit vectors, so recall parity is measured against
a pure-numpy cosine reference (ruling R3): no network, no Chroma — it is
removed in T7 — and no mocks on the read path.
"""
from __future__ import annotations

import hashlib
import math
import os
import random
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
import pytest_asyncio
from qdrant_client import models as qm
from sqlalchemy import create_engine, event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app import database
from app import models as _models  # noqa: F401 — register every table on Base
from app.config import settings
from app.database import Base
from app.models.index_outbox import IndexGeneration
from app.models.memory import Memory
from app.models.user import User
from app.retrieval import vector_backend
from app.retrieval.embedder import EmbeddingDimensionMismatch
from app.retrieval.embedding_fingerprint import (
    canonical_fingerprint,
    fingerprint_generation,
)
from app.retrieval.memory import correction, outbox, vector_store
from app.retrieval.qdrant_filter import build_filter

DIM = 8
GEN_DB = "parity.sqlite"
OTHER_GENERATION = "orivory_memories__othergen"
COLLECTION_NAME_FALLBACK = vector_store.COLLECTION_NAME
# The embedding contract this suite pins (the ambient settings must not decide
# it: the same tests have to hold on a 384-dim lite install and a 1536-dim
# OpenAI one).
FINGERPRINT = {
    "model_id": "test-model",
    "model_revision": "revision-1",
    "dim": DIM,
    "provider": "test",
}


def _fingerprint() -> dict:
    return dict(FINGERPRINT)


def _expected_token() -> str:
    return fingerprint_generation(canonical_fingerprint(FINGERPRINT))


# ── deterministic embeddings (one rule, both faces) ─────────────────────────


def _vector_for(text: str) -> list[float]:
    """Unit vector for a text: same text -> same vector, every run."""
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    raw = [rng.uniform(-1.0, 1.0) for _ in range(DIM)]
    norm = math.sqrt(sum(value * value for value in raw))
    return [value / norm for value in raw]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


async def _fake_embed(texts: list[str]) -> list[list[float]]:
    return [_vector_for(text) for text in texts]


def _fake_embed_sync(texts: list[str]) -> list[list[float]]:
    return [_vector_for(text) for text in texts]


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def env(tmp_path, monkeypatch):
    """Real embedded Qdrant + private SQLite manifest, both on ``tmp_path``.

    Server mode is opt-in FROM THE ENVIRONMENT (``QDRANT_MODE=server``, plus
    ``QDRANT_URL``): the module then drives the live Qdrant the deployment
    uses — the CI integration job points it at the compose service — so the
    real HTTP data path (payload indexes, guard reads) is exercised by the same
    assertions the embedded folder covers. Local stays the default.
    """
    if os.environ.get("QDRANT_MODE") == "server":
        monkeypatch.setattr(settings, "QDRANT_MODE", "server")
        monkeypatch.setattr(
            settings, "QDRANT_URL", os.environ.get("QDRANT_URL", settings.QDRANT_URL)
        )
    else:
        folder = tmp_path / "qdrant"
        folder.mkdir()
        monkeypatch.setattr(settings, "QDRANT_MODE", "local")
        monkeypatch.setattr(settings, "QDRANT_LOCAL_PATH", str(folder))

    url = f"sqlite+aiosqlite:///{tmp_path / GEN_DB}"
    engine = create_async_engine(
        url, connect_args={"check_same_thread": False}, poolclass=NullPool
    )
    event.listen(engine.sync_engine, "connect", database._configure_sqlite_connection)
    sync_engine = create_engine(
        url.replace("+aiosqlite", ""), connect_args={"check_same_thread": False}
    )
    event.listen(sync_engine, "connect", database._configure_sqlite_connection)
    sessions = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    monkeypatch.setattr(database, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(outbox, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(
        database,
        "_get_sync_sessionmaker",
        lambda: sessionmaker(bind=sync_engine, expire_on_commit=False, autoflush=False),
    )
    monkeypatch.setattr(vector_store, "embed_texts", _fake_embed)
    monkeypatch.setattr(vector_store, "embed_texts_sync", _fake_embed_sync)
    # Both guards must see the same contract: embedder binds the fingerprint at
    # import, so patch the alias each module actually calls.
    from app.retrieval import embedder as embedder_module
    from app.retrieval import embedding_fingerprint as fingerprint_module

    for module in (fingerprint_module, embedder_module, vector_store):
        monkeypatch.setattr(module, "current_fingerprint", _fingerprint)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield sessions
    finally:
        await vector_backend.close_clients()
        await engine.dispose()
        sync_engine.dispose()


async def _add_user(sessions, email: str) -> uuid.UUID:
    user_id = uuid.uuid4()
    async with sessions() as db:
        db.add(
            User(
                id=user_id,
                email=email,
                hashed_password="x",
                display_name="Owner",
                is_verified=True,
                is_active=True,
            )
        )
        await db.commit()
    return user_id


@pytest_asyncio.fixture
async def owner(env) -> uuid.UUID:
    """An owner plus the manifest row the SQLite ladder seeds for the
    transitional generation (``app.database._seed_transitional_generation``)."""
    user_id = await _add_user(env, "owner@test.invalid")
    await _activate(env, COLLECTION_NAME_FALLBACK)
    return user_id


async def _activate(sessions, generation: str, fingerprint: str | None = None) -> None:
    """Point the manifest at ``generation`` (one active row per kind)."""
    fingerprint = fingerprint or _expected_token()
    async with sessions() as db:
        rows = (
            await db.execute(select(IndexGeneration).where(IndexGeneration.kind == "memory"))
        ).scalars().all()
        existing = None
        for row in rows:
            row.is_active = False
            if row.generation == generation:
                existing = row
        if existing is not None:
            # A cutover flips the pointer on its own row, never duplicates it.
            existing.is_active = True
            existing.fingerprint = fingerprint
        else:
            db.add(
                IndexGeneration(
                    id=uuid.uuid4().hex,
                    kind="memory",
                    generation=generation,
                    fingerprint=fingerprint,
                    is_active=True,
                )
            )
        await db.commit()


async def _drop_manifest(sessions) -> None:
    async with sessions() as db:
        for row in (await db.execute(select(IndexGeneration))).scalars().all():
            await db.delete(row)
        await db.commit()


def _memory(owner_id: uuid.UUID, *, content: str = "body", **overrides) -> Memory:
    values = {
        "id": uuid.uuid4(),
        "user_id": owner_id,
        "title": "Note",
        "content": content,
        "summary": None,
        "tags": [],
        "salience": 0.5,
        "pinned": False,
        "source_type": "manual_note",
        "source_ref": None,
        "source_url": None,
        "captured_at": datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        "indexed_at": datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        "revision": 1,
        "extra_metadata": {},
    }
    values.update(overrides)
    return Memory(**values)


async def _store(sessions, memories: list[Memory]) -> list[Memory]:
    async with sessions() as db:
        for memory in memories:
            db.add(memory)
        await db.commit()
    return memories


async def _active_name() -> str:
    return (await outbox.active_generation())[0]


async def _payload(memory_id: uuid.UUID) -> dict:
    """One point's persisted payload, read through the real client."""
    client = vector_backend.get_sync_client()
    records = client.retrieve(
        collection_name=await _active_name(), ids=[str(memory_id)], with_payload=True
    )
    assert records, "the point was never written"
    return dict(records[0].payload)


# ── payload contract ────────────────────────────────────────────────────────


async def test_upsert_writes_the_memory_payload_contract(env, owner):
    memory = _memory(owner, tags=["alpha", "beta"], salience=0.25, pinned=True, revision=7)
    await _store(env, [memory])

    vector_store.upsert_memory_sync(memory)

    payload = await _payload(memory.id)
    canonical = canonical_fingerprint(FINGERPRINT)
    assert payload["kind"] == "memory"
    assert payload["user_id"] == str(memory.user_id)
    assert payload["memory_id"] == str(memory.id)
    assert payload["orivory_memory_revision"] == 7
    assert payload["orivory_embed_fingerprint"] == canonical
    assert payload["orivory_embed_generation"] == fingerprint_generation(canonical)
    assert payload["orivory_embed_dim"] == DIM
    assert payload["visibility_state"] == "current"
    assert payload["source_type"] == "manual_note"
    assert payload["captured_at"] == "2026-01-01T12:00:00+00:00"
    assert payload["salience"] == pytest.approx(0.25)
    assert payload["pinned"] is True
    assert payload["tags"] == ["alpha", "beta"]
    assert payload["content"] == "Title: Note\nbody"


async def test_visibility_state_labels_every_state(env, owner):
    current = _memory(owner, content="current")
    needs_check = _memory(
        owner, content="needs-check", extra_metadata={correction.CM_NEEDS_CHECK: True}
    )
    dirty = _memory(owner, content="dirty", extra_metadata={correction.CM_DERIVED_DIRTY: True})
    superseded = _memory(
        owner,
        content="superseded",
        extra_metadata={
            correction.CM_SUPERSEDED_BY: str(uuid.uuid4()),
            correction.CM_DERIVED_DIRTY: True,  # precedence: superseded wins
        },
    )
    await _store(env, [current, needs_check, dirty, superseded])

    vector_store.upsert_memories_sync([current, needs_check, dirty, superseded])

    assert (await _payload(current.id))["visibility_state"] == "current"
    assert (await _payload(needs_check.id))["visibility_state"] == "needs-check"
    assert (await _payload(dirty.id))["visibility_state"] == "dirty"
    assert (await _payload(superseded.id))["visibility_state"] == "superseded"
    # The vector payload is written, never read back as a source of truth: the
    # store still returns the document text it wrote.
    assert (await _payload(current.id))["content"] == "Title: Note\ncurrent"


# ── recall parity against a numpy reference ─────────────────────────────────


async def test_search_scores_match_the_numpy_cosine_reference(env, owner):
    stranger = await _add_user(env, "stranger@test.invalid")
    mine = [_memory(owner, content=f"mine {i}") for i in range(4)]
    foreign = [_memory(stranger, content=f"foreign {i}") for i in range(3)]
    await _store(env, mine + foreign)
    vector_store.upsert_memories_sync(mine + foreign)

    query = _vector_for("a query about notes")
    hits = await vector_store.search_memories(query, user_id=str(owner), top_k=10)

    reference = sorted(
        (
            (
                _cosine(query, _vector_for(vector_store._memory_to_document(memory))),
                str(memory.id),
            )
            for memory in mine
        ),
        key=lambda item: (-item[0], item[1]),
    )
    assert [hit["memory_id"] for hit in hits] == [memory_id for _, memory_id in reference]
    assert [hit["score"] for hit in hits] == pytest.approx(
        [score for score, _ in reference], abs=1e-6
    )

    # The item shape is exactly the Chroma-era one.
    assert set(hits[0]) == {"memory_id", "content", "score", "metadata", "rank", "source"}
    assert hits[0]["source"] == "vector"
    assert [hit["rank"] for hit in hits] == list(range(len(hits)))
    top = next(memory for memory in mine if str(memory.id) == hits[0]["memory_id"])
    assert hits[0]["content"] == vector_store._memory_to_document(top)
    assert hits[0]["metadata"]["kind"] == "memory"
    assert "content" not in hits[0]["metadata"]
    # Tenant boundary: the stranger's vectors are never candidates.
    assert not {hit["memory_id"] for hit in hits} & {str(memory.id) for memory in foreign}


async def test_equal_scores_break_ties_by_memory_id(env, owner):
    """Identical documents score identically: the tie order is stable and
    deterministic (memory_id ascending), never the store's internal order."""
    tied = [_memory(owner, content="identical body") for _ in range(4)]
    await _store(env, tied)
    vector_store.upsert_memories_sync(tied)

    hits = await vector_store.search_memories(
        _vector_for("identical"), user_id=str(owner), top_k=10
    )

    assert len(hits) == 4
    assert len({hit["score"] for hit in hits}) == 1
    assert [hit["memory_id"] for hit in hits] == sorted(str(memory.id) for memory in tied)


async def test_search_on_an_empty_generation_is_empty(env, owner):
    """A fresh generation answers [] — an empty store is not an outage."""
    await _store(env, [_memory(owner, content="indexed elsewhere")])
    await _activate(env, OTHER_GENERATION)

    assert (
        await vector_store.search_memories([0.1] * DIM, user_id=str(owner), top_k=5)
    ) == []


async def test_wrong_dim_query_does_not_bake_its_dim_into_the_generation(env, owner):
    """The read path opens the generation at the CONTRACT dim (the fingerprint),
    never the query's: a wrong-dim query fails the guard instead of creating a
    generation sized to itself."""
    with pytest.raises(EmbeddingDimensionMismatch):
        await vector_store.search_memories(
            [0.1] * (DIM - 3), user_id=str(owner), top_k=5
        )

    generation = await _active_name()
    assert vector_backend.collection_info("memory", generation) == {
        "dim": DIM,
        "distance": "Cosine",
    }


# ── filter parity ───────────────────────────────────────────────────────────


async def test_filters_select_by_payload_fields(env, owner):
    early = datetime(2026, 1, 1, tzinfo=UTC)
    late = datetime(2026, 6, 1, tzinfo=UTC)
    untagged = _memory(owner, content="untagged", tags=[], salience=0.1)
    tagged = _memory(
        owner, content="tagged", tags=["alpha"], salience=0.9, pinned=True, captured_at=late
    )
    tagged_other = _memory(
        owner,
        content="tagged other",
        tags=["beta"],
        salience=0.5,
        source_type="web_clip",
        captured_at=early,
    )
    await _store(env, [untagged, tagged, tagged_other])
    vector_store.upsert_memories_sync([untagged, tagged, tagged_other])

    async def ids(where):
        hits = await vector_store.search_memories(
            _vector_for("query"), user_id=str(owner), top_k=10, where=where
        )
        return {hit["memory_id"] for hit in hits}

    assert await ids({"tags": {"$eq": "alpha"}}) == {str(tagged.id)}
    assert await ids({"tags": {"$in": ["alpha", "beta"]}}) == {
        str(tagged.id),
        str(tagged_other.id),
    }
    assert await ids({"tags": {"$ne": "alpha"}}) == {str(untagged.id), str(tagged_other.id)}
    assert await ids({"tags": {"$nin": ["alpha"]}}) == {str(untagged.id), str(tagged_other.id)}
    # An untagged memory carries no `tags` key: it can never match a tags filter.
    assert str(untagged.id) not in await ids({"tags": {"$in": ["alpha", "beta"]}})
    assert await ids({"pinned": {"$eq": True}}) == {str(tagged.id)}
    assert await ids({"pinned": {"$eq": False}}) == {str(untagged.id), str(tagged_other.id)}
    assert await ids({"source_type": {"$eq": "web_clip"}}) == {str(tagged_other.id)}
    # Numeric range on salience, UTC range on captured_at.
    assert await ids({"salience": {"$gte": 0.5}}) == {str(tagged.id), str(tagged_other.id)}
    assert await ids({"salience": {"$lt": 0.5}}) == {str(untagged.id)}
    assert await ids({"captured_at": {"$gte": late.isoformat()}}) == {str(tagged.id)}
    assert await ids({"captured_at": {"$lt": late.isoformat()}}) == {
        str(untagged.id),
        str(tagged_other.id),
    }
    # A scalar value means $eq.
    assert await ids({"source_type": "manual_note"}) == {str(untagged.id), str(tagged.id)}
    # No where at all: everything the owner owns.
    assert await ids(None) == {str(untagged.id), str(tagged.id), str(tagged_other.id)}


async def test_filters_never_widen_the_tenant_clause(env, owner):
    stranger = await _add_user(env, "stranger@test.invalid")
    mine = _memory(owner, content="mine", tags=["shared"])
    theirs = _memory(stranger, content="theirs", tags=["shared"])
    await _store(env, [mine, theirs])
    vector_store.upsert_memories_sync([mine, theirs])

    hits = await vector_store.search_memories(
        _vector_for("shared"), user_id=str(owner), top_k=10, where={"tags": {"$eq": "shared"}}
    )
    assert [hit["memory_id"] for hit in hits] == [str(mine.id)]

    with pytest.raises(ValueError, match="user_id"):
        await vector_store.search_memories(
            _vector_for("shared"), user_id=str(owner), where={"user_id": {"$eq": str(stranger)}}
        )


def test_build_filter_is_tenant_first_and_allowlisted():
    tenant = qm.FieldCondition(key="user_id", match=qm.MatchValue(value="owner"))
    built = build_filter("owner", {"source_type": {"$eq": "manual_note"}})
    assert built.must[0] == tenant
    assert built.must[1] == qm.FieldCondition(
        key="source_type", match=qm.MatchValue(value="manual_note")
    )

    # No where at all: just the tenant clause.
    bare = build_filter("owner", None)
    assert bare.must == [tenant] and not bare.must_not
    assert build_filter("owner", {}) == bare

    # $ne/$nin cannot be expressed as an include: they land in must_not.
    exclude = build_filter("owner", {"tags": {"$nin": ["spam"]}})
    assert exclude.must == [tenant]
    assert exclude.must_not == [qm.FieldCondition(key="tags", match=qm.MatchAny(any=["spam"]))]

    # Ranges: numeric for salience, datetime for captured_at.
    numeric = build_filter("owner", {"salience": {"$gte": 0.5}})
    assert numeric.must[1] == qm.FieldCondition(key="salience", range=qm.Range(gte=0.5))
    moment = datetime(2026, 1, 1, tzinfo=UTC)
    stamp = build_filter("owner", {"captured_at": {"$gte": moment.isoformat()}})
    assert stamp.must[1] == qm.FieldCondition(key="captured_at", range=qm.DatetimeRange(gte=moment))

    # $contains is list membership for tags (Qdrant matches any element).
    contains = build_filter("owner", {"tags": {"$contains": "alpha"}})
    assert contains.must[1] == qm.FieldCondition(key="tags", match=qm.MatchValue(value="alpha"))


@pytest.mark.parametrize(
    "where, message",
    [
        ({"user_id": {"$eq": "attacker"}}, "user_id"),
        ({"unknown_field": {"$eq": 1}}, "unsupported memory filter"),
        ({"source_type": {"$regex": "x"}}, "unsupported memory filter operator"),
        ({"source_type": {"$eq": "a", "$ne": "b"}}, "one operator"),
        ({"salience": {"$gte": "not-a-number"}}, "salience"),
        ({"captured_at": {"$gte": "not-a-time"}}, "captured_at"),
        ({"source_type": {"$gte": "a"}}, "source_type"),
        ({"tags": {"$gt": 1}}, "tags"),
    ],
)
def test_build_filter_rejects_unknown_key_or_operator(where, message):
    with pytest.raises(ValueError, match=message):
        build_filter("owner", where)


def test_build_filter_rejects_non_object_where():
    with pytest.raises(ValueError, match="object"):
        build_filter("owner", ["not", "a", "dict"])


# ── contract guard ──────────────────────────────────────────────────────────


async def test_populated_generation_without_a_manifest_row_refuses(env, owner):
    memory = _memory(owner)
    await _store(env, [memory])
    vector_store.upsert_memory_sync(memory)

    await _drop_manifest(env)

    with pytest.raises(EmbeddingDimensionMismatch, match="populated"):
        await vector_store.search_memories(_vector_for("q"), user_id=str(owner))
    with pytest.raises(EmbeddingDimensionMismatch, match="populated"):
        vector_store.upsert_memory_sync(memory)
    with pytest.raises(EmbeddingDimensionMismatch, match="populated"):
        vector_store.get_existing_memory_ids_sync([str(memory.id)])


async def test_manifest_fingerprint_mismatch_refuses(env, owner):
    await _activate(env, COLLECTION_NAME_FALLBACK, fingerprint="f" * 64)

    with pytest.raises(EmbeddingDimensionMismatch, match="different embedding contract"):
        await vector_store.search_memories(_vector_for("q"), user_id=str(owner))


async def test_dimension_mismatch_between_query_and_generation_refuses(env, owner):
    memory = _memory(owner)
    await _store(env, [memory])
    vector_store.upsert_memory_sync(memory)

    with pytest.raises(EmbeddingDimensionMismatch, match="dim"):
        await vector_store.search_memories([0.1] * (DIM * 2), user_id=str(owner))


async def test_guard_never_claims_a_generation(env, owner):
    """The read path verifies the manifest, it never writes one (§4.2)."""
    memory = _memory(owner)
    await _store(env, [memory])
    vector_store.upsert_memory_sync(memory)
    async with env() as db:
        before = len((await db.execute(select(IndexGeneration))).scalars().all())

    await vector_store.search_memories(_vector_for("q"), user_id=str(owner))

    async with env() as db:
        after = len((await db.execute(select(IndexGeneration))).scalars().all())
    assert after == before


async def test_an_unreachable_generation_is_a_typed_outage(env, owner, monkeypatch):
    """Acquisition failures stay VectorUnavailableError, never a false []."""
    from app.retrieval.vector_retriever import VectorUnavailableError

    async def _refused(_dim):
        raise ConnectionError("qdrant refused")

    monkeypatch.setattr(vector_store, "_open_collection", _refused)
    with pytest.raises(VectorUnavailableError):
        await vector_store.search_memories(_vector_for("q"), user_id=str(owner))


# ── generation pointer (cutover = a pointer flip) ───────────────────────────


async def test_generation_pointer_flip_moves_reads_to_the_new_collection(env, owner):
    memory = _memory(owner, content="pointer body")
    await _store(env, [memory])
    vector_store.upsert_memory_sync(memory)
    assert await _active_name() == COLLECTION_NAME_FALLBACK
    assert await vector_store.get_memory_ids_present([str(memory.id)]) == {str(memory.id)}

    await _activate(env, OTHER_GENERATION)
    assert await _active_name() == OTHER_GENERATION
    # The pointer moved: the old generation is not even read.
    document = vector_store._memory_to_document(memory)
    assert await vector_store.search_memories(
        _vector_for(document), user_id=str(owner)
    ) == []
    assert await vector_store.get_memory_ids_present([str(memory.id)]) == set()

    vector_store.upsert_memory_sync(memory)  # the backfill writes the new generation
    hits = await vector_store.search_memories(_vector_for(document), user_id=str(owner))
    assert [hit["memory_id"] for hit in hits] == [str(memory.id)]
    # ... and the previous generation still holds the pre-cutover copy.
    client = vector_backend.get_sync_client()
    assert client.retrieve(collection_name=COLLECTION_NAME_FALLBACK, ids=[str(memory.id)])


async def test_async_and_sync_faces_share_one_generation(env, owner):
    first = _memory(owner, content="async write")
    second = _memory(owner, content="sync write")
    await _store(env, [first, second])

    await vector_store.upsert_memory(first)
    vector_store.upsert_memory_sync(second)

    client = vector_backend.get_sync_client()
    for memory in (first, second):
        assert client.retrieve(collection_name=await _active_name(), ids=[str(memory.id)])


# ── delete / presence ───────────────────────────────────────────────────────


async def test_delete_and_presence_helpers(env, owner):
    memories = [_memory(owner, content=f"body {i}") for i in range(3)]
    await _store(env, memories)
    vector_store.upsert_memories_sync(memories)
    ids = [str(memory.id) for memory in memories]

    assert vector_store.get_existing_memory_ids_sync(ids) == set(ids)
    assert await vector_store.get_memory_ids_present(ids) == set(ids)
    assert vector_store.get_existing_memory_ids_sync([]) == set()
    assert await vector_store.get_memory_ids_present([]) == set()

    assert await vector_store.delete_memory(ids[0]) is True
    assert await vector_store.delete_memories(ids[1:2]) is True
    assert vector_store.get_existing_memory_ids_sync(ids) == {ids[2]}
    assert await vector_store.get_memory_ids_present(ids) == {ids[2]}

    vector_store.delete_memories_sync([ids[2]])
    assert await vector_store.get_memory_ids_present(ids) == set()


# ── payload indexes (server mode only) ──────────────────────────────────────


class _RecorderClient:
    """Stand-in for a server client: records the indexes it was asked for."""

    def __init__(self, payload_schema: dict | None = None) -> None:
        self.created: list[tuple[str, object]] = []
        self._schema = dict(payload_schema or {})

    def collection_exists(self, _name: str) -> bool:
        return True

    def get_collection(self, _name: str):
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=qm.VectorParams(size=DIM, distance=qm.Distance.COSINE)
                )
            ),
            payload_schema=dict(self._schema),
        )

    def create_payload_index(self, *, collection_name, field_name, field_schema) -> None:
        self.created.append((field_name, field_schema))
        self._schema[field_name] = field_schema


class _IndexRecorder:
    """The REAL embedded client behind one recorded seam: the real client's
    ``payload_schema`` is where a vacuous check hid, so local mode is pinned
    by counting the calls it must never make."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.index_calls: list[dict] = []

    def create_payload_index(self, **kwargs):
        self.index_calls.append(kwargs)
        return self._inner.create_payload_index(**kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_payload_indexes_are_created_in_server_mode(monkeypatch):
    client = _RecorderClient()
    monkeypatch.setattr(vector_backend, "is_local_mode", lambda: False)
    monkeypatch.setattr(vector_backend, "get_sync_client", lambda: client)

    vector_backend.ensure_collection("memory", OTHER_GENERATION, dim=DIM)

    assert client.created == [
        ("user_id", qm.PayloadSchemaType.KEYWORD),
        ("tags", qm.PayloadSchemaType.KEYWORD),
        ("pinned", qm.PayloadSchemaType.BOOL),
        ("salience", qm.PayloadSchemaType.FLOAT),
        ("captured_at", qm.PayloadSchemaType.DATETIME),
    ]
    assert vector_backend.collection_info("memory", OTHER_GENERATION) == {
        "dim": DIM,
        "distance": "Cosine",
    }


def test_payload_indexes_are_only_added_once(monkeypatch):
    client = _RecorderClient(
        {"user_id": qm.PayloadSchemaType.KEYWORD, "tags": qm.PayloadSchemaType.KEYWORD}
    )
    monkeypatch.setattr(vector_backend, "is_local_mode", lambda: False)
    monkeypatch.setattr(vector_backend, "get_sync_client", lambda: client)

    vector_backend.ensure_collection("memory", OTHER_GENERATION, dim=DIM)

    assert [name for name, _ in client.created] == ["pinned", "salience", "captured_at"]


def test_the_chunk_kind_has_its_own_payload_indexes(monkeypatch):
    """The chunk filters read tenant + conversation + document: server mode
    indexes exactly those fields, never the memory kind's set."""
    client = _RecorderClient()
    monkeypatch.setattr(vector_backend, "is_local_mode", lambda: False)
    monkeypatch.setattr(vector_backend, "get_sync_client", lambda: client)

    vector_backend.ensure_collection("chunk", OTHER_GENERATION, dim=DIM)

    assert client.created == [
        ("user_id", qm.PayloadSchemaType.KEYWORD),
        ("conversation_id", qm.PayloadSchemaType.KEYWORD),
        ("document_id", qm.PayloadSchemaType.KEYWORD),
    ]


def test_an_unknown_kind_gets_no_payload_indexes(monkeypatch):
    client = _RecorderClient()
    monkeypatch.setattr(vector_backend, "is_local_mode", lambda: False)
    monkeypatch.setattr(vector_backend, "get_sync_client", lambda: client)

    vector_backend.ensure_collection("not-a-kind", OTHER_GENERATION, dim=DIM)

    assert client.created == []


async def test_local_mode_creates_no_payload_indexes(monkeypatch, tmp_path):
    """Embedded Qdrant ignores payload indexes (and warns loudly): lite mode
    never asks for one. The assertion counts calls on a REAL client, so it
    fails if the local branch ever called ``create_payload_index``."""
    folder = tmp_path / "qdrant"
    folder.mkdir()
    monkeypatch.setattr(settings, "QDRANT_MODE", "local")
    monkeypatch.setattr(settings, "QDRANT_LOCAL_PATH", str(folder))
    recorder = _IndexRecorder(vector_backend.get_sync_client())
    monkeypatch.setattr(vector_backend, "get_sync_client", lambda: recorder)

    try:
        vector_backend.ensure_collection("memory", OTHER_GENERATION, dim=DIM)
        assert recorder.index_calls == []
        assert recorder.get_collection(OTHER_GENERATION).payload_schema in ({}, None)
    finally:
        await vector_backend.close_clients()


# ── provenance key names (ruling R9) ────────────────────────────────────────


def test_provenance_keys_have_exactly_one_spelling(monkeypatch):
    """The brief's `fingerprint`/`revision` payload names ARE the keys P0/P1a
    persisted — never a second copy under the short name."""
    monkeypatch.setattr(vector_store, "current_fingerprint", _fingerprint)
    memory = _memory(uuid.uuid4(), revision=3)
    metadata = vector_store._memory_to_metadata(memory)

    assert "fingerprint" not in metadata and "revision" not in metadata
    assert metadata["orivory_embed_fingerprint"] == canonical_fingerprint(FINGERPRINT)
    assert metadata["orivory_memory_revision"] == 3
    assert metadata["orivory_embed_generation"] == fingerprint_generation(
        metadata["orivory_embed_fingerprint"]
    )
