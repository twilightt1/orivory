"""Memory + chunk filters -> Qdrant ``Filter`` (spec §4.2, §4.3).

The ONE translation of the caller-facing filter language
(``{"source_type": {"$eq": "note"}}``) into the store's native form, and the
ONE place the tenant clause is built: it is always ``must[0]``, and a caller's
``where`` can never widen it (``user_id`` is rejected outright). The allowlist,
the one-operator-per-field rule and the error types match the Chroma-era
``_build_user_filter`` the API grew around — only the native shape changed.

The chunk family (one collection per generation, payload-filtered) gets its own
builder beside it (ruling R14): tenant first, conversation AND — never a
collection per conversation.

A value that is not an operator object means ``$eq``; ``$ne``/``$nin`` are
exclusions, so they land in ``must_not`` (a point whose field is absent never
matches an include, so an untagged memory is still returned by ``$ne``).
"""
from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

from qdrant_client import models as qm

# The payload fields a caller may filter on. ``user_id`` is deliberately
# absent: it belongs to the authenticated principal, not to the caller.
ALLOWED_FIELDS = frozenset({"source_type", "captured_at", "salience", "pinned", "tags"})
ALLOWED_OPERATORS = frozenset({"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$contains"})

# The only fields a range can be asked of, and the value kind it needs.
_RANGE_FIELDS = {"captured_at": "datetime", "salience": "number"}
_RANGE_OPERATORS = frozenset({"$gt", "$gte", "$lt", "$lte"})
_EXCLUSIONS = frozenset({"$ne", "$nin"})


def build_filter(user_id: str, where: dict[str, Any] | None = None) -> qm.Filter:
    """The immutable tenant clause AND the caller's validated filters."""
    must: list[Any] = [_match("user_id", "$eq", user_id)]
    must_not: list[Any] = []
    if where is None:
        return qm.Filter(must=must)
    if not isinstance(where, dict):
        raise ValueError("memory filters must be an object")

    for key, value in where.items():
        if key == "user_id":
            raise ValueError("user_id is controlled by the authenticated principal")
        if key not in ALLOWED_FIELDS:
            raise ValueError(f"unsupported memory filter: {key}")
        operator, operand = _operator(key, value)
        bucket = must_not if operator in _EXCLUSIONS else must
        bucket.append(_condition(key, operator, operand))
    return qm.Filter(must=must, must_not=must_not or None)


def build_chunk_filter(
    user_id: str | None,
    conversation_id: str,
    *,
    document_id: str | None = None,
) -> qm.Filter:
    """The chunk family's filter: the tenant clause first, the conversation AND.

    Chunks live in ONE collection per generation and are scoped by payload, so
    the tenant clause is what keeps a conversation id — or a document id — from
    reaching another owner's points. ``user_id`` is the authenticated
    principal; ``None`` is the legacy call form, where the conversation scope
    alone is the boundary (an API-face caller must pass it). ``document_id``
    narrows to one document's points (deletes), never widens anything.
    """
    must: list[Any] = []
    if user_id is not None:
        must.append(_match("user_id", "$eq", user_id))
    must.append(_match("conversation_id", "$eq", conversation_id))
    if document_id is not None:
        must.append(_match("document_id", "$eq", document_id))
    return qm.Filter(must=must)


def _operator(key: str, value: Any) -> tuple[str, Any]:
    if not isinstance(value, dict):
        return "$eq", value
    if len(value) != 1:
        raise ValueError(f"memory filter {key} must contain one operator")
    operator, operand = next(iter(value.items()))
    if operator not in ALLOWED_OPERATORS:
        raise ValueError(f"unsupported memory filter operator: {operator}")
    return operator, operand


def _condition(key: str, operator: str, operand: Any) -> qm.FieldCondition:
    if operator in _RANGE_OPERATORS:
        return _range(key, operator, operand)
    if operator in ("$in", "$nin"):
        if not isinstance(operand, (list, tuple, set)):
            raise ValueError(f"memory filter {key} needs a list of values")
        return _match(key, operator, qm.MatchAny(any=list(operand)))
    return _match(key, operator, operand)


def _match(key: str, operator: str, operand: Any) -> qm.FieldCondition:
    """``$eq`` / ``$ne`` / ``$contains`` / ``$in`` / ``$nin``.

    ``$contains`` is list membership: on ``tags`` a MatchValue matches any
    element of the list, which is what a caller means by "contains".
    """
    if isinstance(operand, qm.MatchAny):
        return qm.FieldCondition(key=key, match=operand)
    if isinstance(operand, float):
        # Qdrant matches exactly on keywords/ints/bools only: a float is a
        # range question, and answering it with equality would be a lie.
        raise ValueError(
            f"memory filter {key} cannot use {operator} on a float — use a range operator"
        )
    if isinstance(operand, (bool, int, str)):
        return qm.FieldCondition(key=key, match=qm.MatchValue(value=operand))
    raise ValueError(
        f"memory filter {key} needs a scalar value for {operator}, "
        f"got {type(operand).__name__}"
    )


def _range(key: str, operator: str, operand: Any) -> qm.FieldCondition:
    kind = _RANGE_FIELDS.get(key)
    if kind is None:
        raise ValueError(f"memory filter {key} does not support {operator}")
    bound = operator[1:]  # $gte -> gte
    if kind == "number":
        return qm.FieldCondition(key=key, range=qm.Range(**{bound: _number(key, operand)}))
    return qm.FieldCondition(
        key=key, range=qm.DatetimeRange(**{bound: _timestamp(key, operand)})
    )


def _number(key: str, operand: Any) -> float:
    try:
        return float(operand)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"memory filter {key} needs a number, got {operand!r}") from exc


def _timestamp(key: str, operand: Any) -> datetime:
    """A tz-aware instant. Naive input is rejected: a naive comparison would
    silently shift the window, and the payload stamps UTC (ISO-8601)."""
    if isinstance(operand, datetime):
        parsed = operand
    elif isinstance(operand, date):
        parsed = datetime(operand.year, operand.month, operand.day, tzinfo=UTC)
    elif isinstance(operand, str):
        try:
            parsed = datetime.fromisoformat(operand.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"memory filter {key} needs an ISO-8601 timestamp, got {operand!r}"
            ) from exc
    else:
        raise ValueError(f"memory filter {key} needs an ISO-8601 timestamp, got {operand!r}")
    if parsed.tzinfo is None:
        raise ValueError(f"memory filter {key} needs a timezone-aware timestamp, got {operand!r}")
    return parsed


__all__ = ["ALLOWED_FIELDS", "ALLOWED_OPERATORS", "build_chunk_filter", "build_filter"]
