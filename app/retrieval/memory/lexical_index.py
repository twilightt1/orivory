"""The FTS5 lexical index over ``memories`` — SQLite only (ruling R3(p2)).

``memories`` is the canonical row; ``memory_fts`` is a DERIVED text copy kept by
triggers in the SAME transaction as the memory write (ruling R16(p2)). Identity
never rides the FTS ``rowid``: ``memory_id``/``user_id`` are UNINDEXED columns,
every statement addresses rows by ``memory_id``, and an UPDATE is
delete-by-``memory_id`` followed by an insert — never a rowid-backed upsert.

The virtual table is created by the ladder's ``exec_driver_sql`` DDL
(``database._upgrade_v3_to_v4``), never by ``create_all``: SQLite-only DDL has
no place in model metadata, there is no ORM model for it, and Postgres gets no
FTS at all (no Alembic step). On a non-SQLite deployment this module reports
:class:`LexicalUnavailable` — typed, never a silent "no matches" (ruling R3).

Known limitation, deliberate (spec §7.4): BM25's ``idf``/``avgdl`` are
INDEX-GLOBAL — ``memory_fts`` is one table over every tenant, so another
tenant's corpus shifts this tenant's ordering (reproduced: 300 long foreign
rows flip an intra-tenant order). No authorization impact: the tenant and
visibility clauses are applied BEFORE the LIMIT, so a foreign or superseded
row is never returned — but nothing in the result says the order was computed
against a bigger corpus. The ``score`` :func:`search` returns is the raw
global ``bm25``: T5 fuses by ``rank`` and must never read that score as
tenant-local or as comparable across requests. Measuring the skew is owned by
Task 7's ablation artifact.

Surface (SYNC, like the ladder — an async caller wraps it with
``await session.run_sync(lambda conn: lexical_index.search(conn, ...))``):

- :func:`create_index` — the DDL (virtual table + the three triggers), idempotent.
- :func:`rebuild` — coverage check + backfill; the ladder's transition step and
  the operator's repair path.
- :func:`search` — tenant + namespace + visibility filtered, budgeted MATCH,
  BM25 ascending.
- :func:`is_available` — cheap probe for callers that must fail open (T5's
  vector-outage fallback): ``False`` off SQLite or before the v4 ladder ran.
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from sqlalchemy import (
    Column,
    MetaData,
    String,
    Table,
    Text,
    bindparam,
    literal_column,
    select,
)
from sqlalchemy.engine import Connection

from app.models.memory import Memory
from app.models.types import GUID
from app.retrieval.memory.namespaces import personal_namespace
from app.retrieval.memory.visibility import current_memory_predicate, namespace_predicate

log = logging.getLogger(__name__)

TABLE = "memory_fts"
TOKENIZE = "unicode61 remove_diacritics 2"

# The query budget (R17): a recall query is a sentence, not a document. Both
# caps bound what reaches the FTS tokenizer — a pasted payload or a hostile
# query cannot make the lexical leg the slowest part of a recall.
MAX_QUERY_CHARS = 512
MAX_QUERY_TOKENS = 64

# ``IF NOT EXISTS`` everywhere: the ladder step is re-runnable after a crash
# between the DDL and the ``user_version`` stamp (v3 stays stamped until the
# whole step lands). ponytail: the UPDATE trigger rewrites the row on EVERY
# memory update (recall_count bumps included) — narrow it to
# ``AFTER UPDATE OF title, content, user_id, id`` if write volume ever shows.
CREATE_TABLE = (
    f"CREATE VIRTUAL TABLE IF NOT EXISTS {TABLE} USING fts5("
    f"title, content, memory_id UNINDEXED, user_id UNINDEXED, tokenize='{TOKENIZE}')"
)
CREATE_TRIGGERS = (
    f"CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN "
    f"INSERT INTO {TABLE} (title, content, memory_id, user_id) "
    f"VALUES (new.title, new.content, new.id, new.user_id); END",
    f"CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE ON memories BEGIN "
    f"DELETE FROM {TABLE} WHERE memory_id = old.id; "
    f"INSERT INTO {TABLE} (title, content, memory_id, user_id) "
    f"VALUES (new.title, new.content, new.id, new.user_id); END",
    f"CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN "
    f"DELETE FROM {TABLE} WHERE memory_id = old.id; END",
)

# A standalone table object: the virtual table must NOT live on ``Base.metadata``
# (create_all would try to CREATE TABLE it). It exists only to be joined.
_FTS = Table(
    TABLE,
    MetaData(),
    Column("title", Text),
    Column("content", Text),
    Column("memory_id", String),
    Column("user_id", String),
)

# Literal tokens: every FTS5 operator character (``"``, ``*``, ``-``, ``^``,
# ``:``, parens) is a separator here, and each token is re-emitted QUOTED — the
# user's text is data, never grammar. ``NEAR``/``OR``/``NOT`` survive as quoted
# words, so they match those words instead of switching the query language on.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class LexicalUnavailable(RuntimeError):
    """No lexical index on this deployment (ruling R3(p2)).

    FTS5 is SQLite-only: a Postgres install has no ``memory_fts`` and never
    gets one. The caller (recall's lexical leg / the vector-outage fallback)
    counts this and answers without the lexical leg — it must never be
    translated into an empty result set, which reads as "nothing matched".

    Declared for callers and tests: no production raise site of its own, since
    every serving path gates on :func:`is_available` / the dialect first — the
    raise in :func:`search` / :func:`rebuild` is a contract guard, not a path
    an operator can reach.
    """


def is_available(conn: Connection) -> bool:
    """True when ``conn`` has a usable lexical index (SQLite + the v4 ladder).

    Answers without touching a non-SQLite connection, so a Postgres caller can
    branch on it cheaply instead of catching :class:`LexicalUnavailable`.
    """
    if conn.dialect.name != "sqlite":
        return False
    row = conn.exec_driver_sql(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (TABLE,)
    ).first()
    return row is not None


def create_index(conn: Connection) -> None:
    """Create the virtual table and its triggers (idempotent DDL)."""
    conn.exec_driver_sql(CREATE_TABLE)
    for trigger in CREATE_TRIGGERS:
        conn.exec_driver_sql(trigger)


def coverage(conn: Connection) -> dict[str, int]:
    """Canonical-vs-index coverage, BY ID — not just a row count.

    Two drifts cancel out in a count comparison (a duplicated index row for
    every missing one), so the check is the two set differences. Runs only on
    the ladder/repair path, never on the read path.
    """
    canonical = conn.exec_driver_sql("SELECT count(*) FROM memories").scalar_one()
    indexed = conn.exec_driver_sql(f"SELECT count(*) FROM {TABLE}").scalar_one()
    missing = conn.exec_driver_sql(
        f"SELECT count(*) FROM memories m WHERE NOT EXISTS "
        f"(SELECT 1 FROM {TABLE} f WHERE f.memory_id = m.id)"
    ).scalar_one()
    orphan = conn.exec_driver_sql(
        f"SELECT count(*) FROM {TABLE} f WHERE NOT EXISTS "
        f"(SELECT 1 FROM memories m WHERE m.id = f.memory_id)"
    ).scalar_one()
    return {"canonical": int(canonical), "indexed": int(indexed),
            "missing": int(missing), "orphan": int(orphan)}


def rebuild(conn: Connection) -> dict[str, Any]:
    """Backfill the index from ``memories`` when coverage drifted.

    Drift is a missing row, an orphan row, OR a duplicate one: the duplicate
    leaves ``missing``/``orphan`` at zero and only moves the counts — and left
    in place it is returned twice by the DISTINCT-free join in :func:`search`.
    The ladder calls this once, on the v3 -> v4 transition; an operator calls it
    to repair drift (a write that bypassed the triggers, a hand-edited file).
    A healthy index is left untouched — report ``rebuilt: False`` and return.
    """
    if conn.dialect.name != "sqlite":
        raise LexicalUnavailable(
            f"the lexical index ({TABLE}) is SQLite-only (ruling R3(p2)); "
            f"this connection is {conn.dialect.name}"
        )
    before = coverage(conn)
    rebuilt = bool(before["missing"] or before["orphan"]
                   or before["indexed"] != before["canonical"])
    if rebuilt:
        log.warning("Rebuilding the FTS5 memory index", extra=before)
        conn.exec_driver_sql(f"DELETE FROM {TABLE}")
        conn.exec_driver_sql(
            f"INSERT INTO {TABLE} (title, content, memory_id, user_id) "
            f"SELECT title, content, id, user_id FROM memories"
        )
    return {**before, "rebuilt": rebuilt, "indexed_after": coverage(conn)["indexed"]}


def match_expression(query: str) -> str:
    """The user's text as an FTS5 MATCH expression of literal phrases.

    Bounded by :data:`MAX_QUERY_CHARS` / :data:`MAX_QUERY_TOKENS`; an empty
    result (punctuation-only input) means "nothing to match" — the caller must
    not run MATCH with it, SQLite rejects an empty expression.
    """
    text = query[:MAX_QUERY_CHARS]
    tokens = _TOKEN_RE.findall(text)
    if len(tokens) > MAX_QUERY_TOKENS:
        log.debug("Truncating the lexical query", extra={"tokens": len(tokens)})
        tokens = tokens[:MAX_QUERY_TOKENS]
    return " ".join(f'"{token}"' for token in tokens)


def search(conn: Connection, query: str, *, user_id: uuid.UUID | str,
           limit: int, namespace: str | None = None) -> list[dict[str, Any]]:
    """Rank ``user_id``'s CURRENT memories against ``query`` (best first).

    Returns ``[{"memory_id": <canonical uuid str>, "score": <bm25>, "rank": i}]``
    with zero-based ranks — BM25 ascending is better, so the lexical order is
    dense-independent and ready for rank fusion (spec §7.4).

    The tenant, namespace and ``current_memory_predicate()`` clauses are applied
    on the canonical join BEFORE the LIMIT: a superseded, dirty, foreign-tenant
    or out-of-namespace row can never occupy one of the ``limit`` slots,
    whatever its BM25 says. The namespace clause is exact here — ``memory_fts``
    is not a second copy of the boundary, the join reads ``memories.namespace``,
    and the ladder backfilled every pre-P4 row to ``personal`` (so R32's
    "a missing key is personal" branch has no analogue in this leg).

    ``namespace`` defaults to the caller's own (``namespaces.personal_namespace``)
    through ``visibility.namespace_predicate`` — the one SQL spelling of the
    boundary, so this leg cannot drift from the rows (P4a: personal-only).
    """
    if conn.dialect.name != "sqlite":
        # Before the empty-query short circuit, deliberately: off SQLite this is
        # "unavailable", NOT "no matches" (ruling R3).
        raise LexicalUnavailable(
            f"the lexical index ({TABLE}) is SQLite-only (ruling R3(p2)); "
            f"this connection is {conn.dialect.name}"
        )
    expression = match_expression(query)
    if not expression or limit <= 0:
        return []
    if namespace is None:
        namespace = personal_namespace(user_id)

    tenant = bindparam("tenant_id", user_id, type_=GUID())
    stmt = (
        select(Memory.id.label("memory_id"),
               literal_column(f"bm25({TABLE})").label("score"))
        .select_from(_FTS.join(Memory, Memory.id == _FTS.c.memory_id))
        .where(literal_column(TABLE).op("MATCH")(bindparam("match", expression)))
        .where(_FTS.c.user_id == tenant, Memory.user_id == tenant)
        .where(namespace_predicate(namespace))
        .where(current_memory_predicate())
        .order_by(literal_column(f"bm25({TABLE})").asc(), Memory.id.asc())
        .limit(limit)
    )
    rows = conn.execute(stmt).all()
    return [{"memory_id": str(row.memory_id), "score": float(row.score), "rank": rank}
            for rank, row in enumerate(rows)]


__all__ = [
    "CREATE_TABLE",
    "CREATE_TRIGGERS",
    "LexicalUnavailable",
    "MAX_QUERY_CHARS",
    "MAX_QUERY_TOKENS",
    "TABLE",
    "coverage",
    "create_index",
    "is_available",
    "match_expression",
    "rebuild",
    "search",
]
