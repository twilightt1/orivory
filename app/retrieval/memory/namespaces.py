"""Namespace values and the personal default (P4a).

A namespace is an authorization boundary, not a label: two memories with the
same text in two namespaces are two facts with different owners' permissions.
This phase ships the schema and the only value there is — ``personal`` — while
team sharing stays OFF (``sharing`` is deferred to P4b):

    - ``PERSONAL`` is THE spelling. It is what the SQLite ladder backfills into
      ``memories.namespace`` and what the model declares as its server default,
      and it is what every predicate (``visibility.namespace_predicate``) is
      built from — spelled once here, so a predicate can never drift from the
      value the rows actually carry.
    - ``personal_namespace(user_id)`` answers the namespace a user's own rows
      live in. In P4a that is a constant for every user; the argument exists
      because a later phase may key it per user, and callers should already be
      routing through it rather than writing the literal.
    - ``namespace_of(row)`` answers the namespace of a row that may predate the
      column (a pre-P4 payload, a detached object): absent means personal.

Namespace is never derived from client input in this phase, so nothing here
reads a request.
"""
from __future__ import annotations

from typing import Any

PERSONAL = "personal"

__all__ = ["PERSONAL", "namespace_of", "personal_namespace"]


def personal_namespace(user_id: Any) -> str:
    """The namespace ``user_id``'s own memories live in (P4a: always ``personal``).

    The argument is unused on purpose: P4a has exactly one namespace, and the
    interface is per-user so a later phase can key it without touching callers.
    """
    return PERSONAL


def namespace_of(row: Any) -> str:
    """The namespace of ``row``; a row without one (pre-P4) is personal."""
    return getattr(row, "namespace", None) or PERSONAL
