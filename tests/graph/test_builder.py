"""The graph builder's DB contract: no lock across the LLM call, no lost race.

Each test runs the REAL builder against a private SQLite file with the LLM
client stubbed (``tests/graph/conftest.py``), so the transaction boundaries,
the unique-key backstops and the processed-marker semantics are the ones a
production boot gets.
"""
from __future__ import annotations

import sqlite3
import uuid

from sqlalchemy import func, select

from app.graph.builder import build_memory_graph_sync
from app.models.entity import Entity, MemoryEntity
from app.models.memory import Memory
from tests.graph.conftest import add_memory, stub_llm

ENTITY_PAYLOAD = '{"entities": [{"name": "Alice", "type": "person", "salience": 0.9}]}'
#: id 288's pin needs the SECOND provider call: ``extract_relations`` answers
#: early below two entities, so a one-entity payload never reaches the round
#: trip that used to run with the SQLite write lock already held.
PAIR_PAYLOAD = (
    '{"entities": [{"name": "Alice", "type": "person", "salience": 0.9},'
    ' {"name": "Bob", "type": "person", "salience": 0.7}]}'
)
RELATION_PAYLOAD = '{"relations": []}'


def _entity_rows(store) -> list[Entity]:
    with store.maker() as session:
        return list(session.execute(select(Entity)).scalars().all())


def _memory_metadata(store, memory_id: uuid.UUID) -> dict:
    with store.maker() as session:
        memory = session.get(Memory, memory_id)
        return dict(memory.extra_metadata or {})


def _second_writer(path: str, memory_hex: str) -> str | None:
    """Write to the store the way the request path does, with a SHORT wait.

    Returns the error text when the write could not land: the point is that a
    writer must not be queued behind an open LLM round trip (the production
    busy_timeout is 5 s, so a slower extraction is a failed request).
    """
    conn = sqlite3.connect(path, timeout=0.5)
    try:
        conn.execute(
            "UPDATE memories SET recall_count = recall_count + 1 WHERE id = ?", (memory_hex,)
        )
        conn.commit()
    except sqlite3.OperationalError as exc:
        return str(exc)
    finally:
        conn.close()
    return None


def test_no_write_lock_is_held_across_the_extraction_call(graph_store, monkeypatch):
    """id 288: the entity flush used to sit before ``extract_relations``.

    SQLite takes the write lock at that flush and holds it until the commit
    AFTER the LLM round trip: every other writer waits out ``busy_timeout``
    and then fails ``database is locked``. Collect first, call the provider,
    then persist.
    """
    probes: list[tuple[int, str | None]] = []
    stub_llm(
        monkeypatch,
        [PAIR_PAYLOAD, RELATION_PAYLOAD],
        on_call=lambda call: probes.append(
            (call, _second_writer(graph_store.path, graph_store.memory_id_hex))
        ),
    )

    with graph_store.maker() as session:
        result = build_memory_graph_sync(session, str(graph_store.memory_id))

    assert result.entities_created == 2
    # Two calls, not one: the RELATIONS call is the one that used to run with
    # the write lock held. A one-entity payload made this test pass on the
    # broken code because ``extract_relations`` returned before the call.
    assert [call for call, _ in probes] == [1, 2], (
        f"the second provider call did not happen: {probes}"
    )
    assert all(error is None for _, error in probes), (
        f"a second writer was blocked while the extraction ran: {probes}"
    )


def test_a_concurrent_same_name_insert_converges_on_the_existing_row(graph_store, monkeypatch):
    """ids 286/78: select-then-insert with no conflict path.

    A rival build that lands the same (user, name, type) between our SELECT and
    our INSERT used to abort this build with IntegrityError — the memory then
    kept NO graph (the failure is only logged). It must converge on the row the
    rival wrote.
    """
    rival_id = uuid.uuid4()
    with graph_store.maker() as session:  # the rival build won the race, committed
        session.add(Entity(
            id=rival_id, user_id=graph_store.user_id, name="Alice", entity_type="person",
            aliases=[], description=None, mention_count=1, extra_metadata={},
        ))
        session.commit()
    stub_llm(monkeypatch, [ENTITY_PAYLOAD, RELATION_PAYLOAD])

    with graph_store.maker() as session:
        real_execute = session.execute
        stale = {"pending": True}

        class _StaleRead:
            def scalar_one_or_none(self):
                return None

            def scalars(self):
                return self

            def first(self):
                return None  # what the losing SELECT saw a moment earlier

        def stale_execute(statement, *args, **kwargs):
            if stale["pending"] and "FROM entities" in str(statement):
                stale["pending"] = False
                return _StaleRead()
            return real_execute(statement, *args, **kwargs)

        monkeypatch.setattr(session, "execute", stale_execute)
        result = build_memory_graph_sync(session, str(graph_store.memory_id))

    assert result.entities_created == 0, "the rival's row is THE row"
    rows = _entity_rows(graph_store)
    assert [row.id for row in rows] == [rival_id]
    with graph_store.maker() as session:
        links = session.execute(
            select(func.count()).select_from(MemoryEntity).where(MemoryEntity.entity_id == rival_id)
        ).scalar_one()
    assert links == 1, "the memory must still be linked to the entity it shares"
    assert _memory_metadata(graph_store, graph_store.memory_id).get("graph_extracted_at")


def test_legacy_case_variant_rows_do_not_raise_multiple_results(graph_store, monkeypatch):
    """id 78: the unique key is on the raw name, the lookup on ``lower(name)``.

    Both spellings could therefore exist, and the next build raised
    ``MultipleResultsFound`` for the memory — its whole graph was never written.
    The lookup must resolve to ONE row deterministically.
    """
    with graph_store.maker() as session:
        session.add(Entity(user_id=graph_store.user_id, name="Mom", entity_type="person",
                           aliases=[], description=None, mention_count=0, extra_metadata={}))
        session.commit()
    with graph_store.maker() as session:
        session.add(Entity(user_id=graph_store.user_id, name="mom", entity_type="person",
                           aliases=[], description=None, mention_count=0, extra_metadata={}))
        session.commit()
    stub_llm(monkeypatch, ['{"entities": [{"name": "MOM", "type": "person"}]}', RELATION_PAYLOAD])

    with graph_store.maker() as session:
        result = build_memory_graph_sync(session, str(graph_store.memory_id))

    assert result.error is None
    assert result.entities_created == 0
    assert len(_entity_rows(graph_store)) == 2, "the build reuses a row, never mints a third"


def test_a_case_variant_extraction_reuses_the_entity(graph_store, monkeypatch):
    """id 78 (write side): 'Mom' from one memory and 'mom' from another is ONE row."""
    stub_llm(monkeypatch, ['{"entities": [{"name": "Mom", "type": "person"}]}', RELATION_PAYLOAD])
    with graph_store.maker() as session:
        first = build_memory_graph_sync(session, str(graph_store.memory_id))
    assert first.entities_created == 1

    second_memory = add_memory(graph_store, content="Mom called about the trip.")
    stub_llm(monkeypatch, ['{"entities": [{"name": "mom", "type": "person"}]}', RELATION_PAYLOAD])
    with graph_store.maker() as session:
        second = build_memory_graph_sync(session, str(second_memory))

    assert second.error is None
    assert second.entities_created == 0
    assert [row.name for row in _entity_rows(graph_store)] == ["Mom"]


def test_a_wrong_typed_entities_payload_leaves_the_memory_rebuildable(graph_store, monkeypatch):
    """id 363: a parseable payload whose ``entities`` is not a list.

    It was coerced to ``[]`` and returned as a SUCCESS, so the builder stamped
    ``graph_extracted_at`` and nothing ever rebuilt the memory: it stayed
    graph-less, silently, forever. It is a schema error — the fallback runs and
    the memory keeps its rebuild eligibility.
    """
    stub_llm(monkeypatch, ['{"entities": {"name": "Alice"}}', RELATION_PAYLOAD])

    with graph_store.maker() as session:
        result = build_memory_graph_sync(session, str(graph_store.memory_id))

    assert result.fallback_used is True
    assert result.entities_extracted >= 1, "the deterministic fallback still extracts"
    metadata = _memory_metadata(graph_store, graph_store.memory_id)
    assert metadata.get("graph_entity_error"), "the schema failure must be visible"
    assert not metadata.get("graph_extracted_at"), (
        "a failed extraction must not be stamped as processed: nothing would rebuild it"
    )


def test_a_wrong_typed_relations_payload_leaves_the_memory_rebuildable(graph_store, monkeypatch):
    """id 363's sibling: a parseable payload whose ``relations`` is not a list.

    It was coerced to ``[]`` and returned as SUCCESS, so the builder stamped
    ``graph_extracted_at`` on a memory carrying only the deterministic fallback
    relations — the same failure mode as ``entities``, one field over.
    """
    stub_llm(monkeypatch, [PAIR_PAYLOAD, '{"relations": {"source": "Alice"}}'])

    with graph_store.maker() as session:
        result = build_memory_graph_sync(session, str(graph_store.memory_id))

    assert result.fallback_used is True
    assert result.relations_extracted >= 1, "the deterministic fallback still runs"
    metadata = _memory_metadata(graph_store, graph_store.memory_id)
    assert metadata.get("graph_relation_error"), "the schema failure must be visible"
    assert not metadata.get("graph_extracted_at"), (
        "a failed extraction must not be stamped as processed: nothing would rebuild it"
    )
