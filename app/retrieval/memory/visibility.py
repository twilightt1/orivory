"""One SQL visibility rule mirroring ``correction.state_of`` (spec §4.2).

``state_of`` stays the Python authority — it labels rows already in memory.
These expressions answer the same question in SQL so a reader can filter and
label *before* its own LIMIT: a post-hoc Python filter after a ``limit``
silently returns fewer rows than asked for, or none at all.

Vocabulary (precedence superseded > dirty > needs-check > current):

- current / needs-check — served, labeled with their state.
- superseded — readable in direct get / timeline / history views, labeled.
- dirty — never served, never used as context or rerank evidence.

The namespace boundary (``namespace_predicate``) is the second rule every reader
of ``memories`` composes: an authorization predicate, not a lifecycle label —
two rows with the same text in two namespaces are two facts with different
owners' permissions (spec §8.2). It belongs in the SAME statement as the row it
protects, for the same reason the lifecycle predicates do.
"""
from __future__ import annotations

from sqlalchemy import and_, case

from app.models.memory import Memory
from app.retrieval.memory.correction import (
    CM_DERIVED_DIRTY,
    CM_NEEDS_CHECK,
    CM_SUPERSEDED_BY,
)


def namespace_predicate(namespace: str):
    """The namespace boundary as a SQL predicate — the one spelling of it.

    ``namespace`` always comes from ``app.retrieval.memory.namespaces`` (never
    from client input, never a literal in a query), so a predicate cannot drift
    from the value the rows actually carry. Compose it with the lifecycle
    predicate: ``where(Memory.user_id == ..., namespace_predicate(ns), not_dirty_predicate())``.

    ``None``/empty is REFUSED, like ``qdrant_filter.build_filter`` refuses it:
    ``None`` would compile to ``namespace IS NULL`` and ``''`` to ``= ''`` —
    predicates no caller means. ``None`` means the whole database in the
    migration CLI's vocabulary, so compiling it into a WHERE clause would read
    rows the caller never authorized. A padded value is normalized once
    (``str(namespace).strip()``) to the spelling the rows carry, for the same
    reason: matched verbatim it would match nothing, in silence.
    """
    if namespace is None or not str(namespace).strip():
        raise ValueError(
            "namespace_predicate needs a namespace: it is an authorization boundary — "
            "None compiles to IS NULL (the CLI's whole-DB vocabulary), never to a "
            "predicate a reader may run"
        )
    return Memory.namespace == str(namespace).strip()


def _has(key: str):
    """The cm_* marker as a string-or-NULL SQL expression (absent -> NULL).

    A presence check, not truthiness: every writer stores truthy markers
    (``True`` / an id string), which is how ``state_of`` reads them too.
    """
    return Memory.extra_metadata[key].as_string().is_not(None)


def not_dirty_predicate():
    """List-view visibility: a dirty (stale derived) row is never served."""
    return ~_has(CM_DERIVED_DIRTY)


def current_memory_predicate():
    """Rows ``state_of`` calls current or needs-check.

    Neither superseded (history, readable elsewhere) nor dirty (wrong) — apply
    it before any LIMIT so stale rows cannot crowd a reader's slice.
    """
    return and_(~_has(CM_SUPERSEDED_BY), ~_has(CM_DERIVED_DIRTY))


def state_expression():
    """SELECT-side state label; precedence mirrors ``state_of`` exactly."""
    return case(
        (_has(CM_SUPERSEDED_BY), "superseded"),
        (_has(CM_DERIVED_DIRTY), "dirty"),
        (_has(CM_NEEDS_CHECK), "needs-check"),
        else_="current",
    )
