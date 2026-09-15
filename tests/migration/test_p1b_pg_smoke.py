"""P1b Postgres-mode cutover smoke (ruling R45).

The SQLite ladder is not what a server deployment runs: Postgres has no
``user_version`` ladder, so the CLI's ``_expand`` writes the two generation rows
DIRECTLY through ``database.activate_generations`` — the one definition of "the
active generation is memory + chunk at the current contract" that the SQLite
ladder shares (ruling R10). Nothing about that path can be exercised on a
SQLite file, and the whole point of the ruling is that it must not rot.

Skip-when-absent by construction: without a Postgres ``DATABASE_URL`` (or with
services down) the module skips wholesale, so the infra-free CI steps stay
green. Locally: ``docker compose up -d postgres`` (host port from
``POSTGRES_PORT``, 55432 by default) then

    DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:55432/ragdb \\
        python -m pytest --confcutdir=tests/migration \\
        tests/migration/test_p1b_pg_smoke.py -q

The vector side is an embedded Qdrant folder (no server needed): the claim under
test is the PG pointer path, and the SQLite suites already own the Qdrant
readback. Everything unsupported here is deleted again in teardown, so a shared
test database is left as it was found.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import sessionmaker

from app import database
from app import models as _models  # noqa: F401 — register every table on Base
from app.config import settings
from app.database import Base
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.index_outbox import IndexGeneration
from app.models.memory import Memory
from app.models.user import User
from app.retrieval import vector_backend
from app.retrieval.embedding_fingerprint import (
    canonical_fingerprint,
    fingerprint_generation,
    generation_name,
)
from app.retrieval.memory import outbox, vector_store

pytestmark = pytest.mark.skipif(
    not settings.DATABASE_URL.startswith("postgresql"),
    reason=("the Postgres cutover path needs a Postgres DATABASE_URL "
            "(docker compose up -d postgres; POSTGRES_PORT configures the host port)"),
)

# The contract this smoke pins — never the ambient one (the same test has to
# hold whether the deployment embeds at 384 or 1536 dims).
DIM = 8
FINGERPRINT = {
    "model_id": "pg-smoke-model",
    "model_revision": "revision-1",
    "dim": DIM,
    "provider": "test",
}


def _fingerprint() -> dict:
    return dict(FINGERPRINT)


def _vector(text: str) -> list[float]:
    import hashlib
    import math
    import random

    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    raw = [rng.uniform(-1.0, 1.0) for _ in range(DIM)]
    norm = math.sqrt(sum(value * value for value in raw))
    return [value / norm for value in raw]


def _fake_embed_sync(texts: list[str]) -> list[list[float]]:
    return [_vector(text) for text in texts]


@pytest.fixture
def pg(tmp_path, monkeypatch, migrate_cli):
    """A real Postgres database (schema ensured) + a temp embedded Qdrant."""
    folder = tmp_path / "qdrant"
    folder.mkdir()
    # Server deployments keep the CLI's sidecars in the working directory: run
    # from a temp cwd so the marker/lock/checkpoint never land in the repo.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "QDRANT_MODE", "local")
    monkeypatch.setattr(settings, "QDRANT_LOCAL_PATH", str(folder))
    monkeypatch.setattr(settings, "APP_PORT", 1)  # nothing listens: quiesced

    from app.retrieval import embedder as embedder_module
    from app.retrieval import embedding_fingerprint as fingerprint_module
    from app.retrieval import vector_retriever

    for module in (fingerprint_module, embedder_module, vector_store, vector_retriever,
                   migrate_cli):
        monkeypatch.setattr(module, "current_fingerprint", _fingerprint)
    monkeypatch.setattr(vector_store, "embed_texts_sync", _fake_embed_sync)
    monkeypatch.setattr(vector_retriever, "embed_texts_sync", _fake_embed_sync)
    monkeypatch.setattr(migrate_cli, "embed_texts_sync", _fake_embed_sync)

    # The app-level sync engine is cached per URL: this run's DATABASE_URL wins.
    database.get_sync_engine.cache_clear()
    database._get_sync_sessionmaker.cache_clear()

    url = settings.DATABASE_URL.replace("+asyncpg", "+psycopg2")
    engine = create_engine(url, pool_pre_ping=True)
    Base.metadata.create_all(engine)  # idempotent: CI's migrate step already did it
    sessions = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)

    user_id, memory_id, conversation_id, document_id, chunk_id = (
        uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    )
    memory_generation = generation_name("memory")
    chunk_generation = generation_name("chunk")
    with sessions() as db:
        db.add(User(id=user_id, email=f"{user_id.hex}@pg-smoke.invalid",
                    hashed_password="x", display_name="PG smoke",
                    is_verified=True, is_active=True))
        db.add(Memory(id=memory_id, user_id=user_id, content="pg smoke memory", tags=[]))
        db.add(Conversation(id=conversation_id, user_id=user_id, document_count=1))
        db.add(Document(id=document_id, conversation_id=conversation_id,
                        filename="pg-smoke.txt", file_path="uploads/pg-smoke.txt",
                        chunk_count=1))
        db.add(DocumentChunk(
            id=chunk_id, document_id=document_id, content="pg smoke chunk",
            chunk_index=0, revision=1,
            chunk_metadata={"conversation_id": str(conversation_id),
                            "document_id": str(document_id), "chunk_type": "child"},
        ))
        db.commit()

    try:
        yield SimpleNamespace(
            tmp_path=tmp_path, folder=folder, sessions=sessions, engine=engine,
            cli=migrate_cli, user_id=user_id, memory_id=memory_id, chunk_id=chunk_id,
            memory_generation=memory_generation, chunk_generation=chunk_generation,
        )
    finally:
        import asyncio

        asyncio.run(vector_backend.close_clients())
        with sessions() as db:
            db.execute(delete(IndexGeneration).where(
                IndexGeneration.generation.in_([memory_generation, chunk_generation])))
            db.execute(delete(DocumentChunk).where(DocumentChunk.id == chunk_id))
            db.execute(delete(Document).where(Document.id == document_id))
            db.execute(delete(Conversation).where(Conversation.id == conversation_id))
            db.execute(delete(Memory).where(Memory.id == memory_id))
            db.execute(delete(User).where(User.id == user_id))
            db.commit()
        engine.dispose()


def _generations(pg) -> dict[tuple[str, str], IndexGeneration]:
    with pg.sessions() as db:
        return {
            (row.kind, row.generation): row
            for row in db.execute(select(IndexGeneration)).scalars().all()
        }


def test_pg_expand_backfill_and_cutover_flip_the_pointer(pg):
    """The Postgres path, end to end: expand writes INACTIVE rows, the pointer
    moves only at cutover, and the runtime's own read of the manifest follows."""
    cli = pg.cli

    # ── the expand (R10/R24): both rows exist, NEITHER is active. An
    # un-migrated PG deployment keeps serving its transitional pointer, so the
    # read path fails loud instead of answering zero hits from an empty
    # generation nobody built.
    cli._expand()
    after_expand = _generations(pg)
    assert set(after_expand) == {
        ("memory", pg.memory_generation), ("chunk", pg.chunk_generation)}
    assert not any(row.is_active for row in after_expand.values())
    assert cli.active_generation_name(pg.sessions(), "memory") is None
    assert outbox.active_generation_sync()[0] == vector_store.COLLECTION_NAME

    # ── the backfill: keyset by primary key, into the named generation.
    memory_report = cli.backfill(kind="memory", batch=2)
    chunk_report = cli.backfill(kind="chunk", batch=2)
    assert (memory_report["generation"], memory_report["upserted"]) == (pg.memory_generation, 1)
    assert (chunk_report["generation"], chunk_report["upserted"]) == (pg.chunk_generation, 1)
    assert cli.verify(kind="memory")["ok"] is True
    assert cli.verify(kind="chunk")["ok"] is True

    # ── the flip: one transaction, both kinds, gated on the audit.
    report = cli.cutover(yes=True)
    assert report["active"] == {"memory": pg.memory_generation, "chunk": pg.chunk_generation}
    assert report["previous"]["memory"] == vector_store.COLLECTION_NAME

    token = fingerprint_generation(canonical_fingerprint(_fingerprint()))
    active = {row.kind: row for row in _generations(pg).values() if row.is_active}
    assert {kind: row.generation for kind, row in active.items()} == report["active"]
    assert {row.fingerprint for row in active.values()} == {token}

    # The RUNTIME reads the same manifest through its own session maker: the
    # flipped pointer is what a request path would serve — for both kinds.
    assert outbox.active_generation_sync() == (pg.memory_generation, token)
    assert outbox.active_generation_sync(kind="chunk") == (pg.chunk_generation, token)
    assert cli.verify(kind="memory")["ok"] is True

    # The runtime's readback agrees with SQL, on the Postgres-side rows too.
    assert str(pg.memory_id) in _scroll(pg.memory_generation)
    assert str(pg.chunk_id) in _scroll(pg.chunk_generation)


def _scroll(generation: str) -> set[str]:
    points, _ = vector_backend.get_sync_client().scroll(
        collection_name=generation, limit=64, offset=None,
        with_payload=False, with_vectors=False,
    )
    return {str(point.id) for point in points}
