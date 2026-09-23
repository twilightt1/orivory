"""`memory_rows` must mark EVERY non-servable state ineligible.

The migration CLI's backfill and its `verify` gate decide eligibility with one
question — `record["reason"] is None` — so a state that maps to no reason is
silently re-embedded and accepted after cutover. Review finding, verified:
`state_of` has four values plus needs-check, and `memory_rows` mapped only
`superseded`, `dirty` and `suppressed`; an INVALIDATED row (a forgotten
memory — `state_of`'s top precedence) walked through as eligible, so a re-run
would put erased content back into the store the user was told was cleared.
"""
from __future__ import annotations

import datetime
import importlib.util
import uuid
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import models  # noqa: F401 — register every model on Base
from app.database import Base
from app.models import Memory
from app.retrieval.memory.correction import (
    CM_DERIVED_DIRTY,
    CM_INVALIDATED,
    CM_SUPERSEDED_BY,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "scripts" / "migrate_qdrant.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("migrate_qdrant", CLI_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _memory(title: str, **meta) -> Memory:
    return Memory(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        title=title,
        content=title,
        source_type="manual_note",
        source_ref=None,
        tags=[],
        captured_at=datetime.datetime.now(datetime.UTC),
        extra_metadata=dict(meta),
    )


def _reasons(*memories: Memory) -> dict[str, str | None]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(memories)
        session.commit()
        return {
            memory_id: record["reason"]
            for memory_id, record in _load_cli().memory_rows(session).items()
        }


def test_an_invalidated_row_is_not_eligible_for_backfill():
    forgotten = _memory("forgotten", **{CM_INVALIDATED: True})
    current = _memory("current")

    reasons = _reasons(forgotten, current)
    assert reasons[str(forgotten.id)] == "invalidated", (
        "an invalidated row must carry a reason: `reason is None` is the only "
        "eligibility test the backfill and the cutover verify share"
    )
    assert reasons[str(current.id)] is None


def test_invalidated_outranks_superseded_and_dirty_as_a_reason():
    """state_of's precedence, mirrored in the reason map."""
    both = _memory("superseded and invalidated",
                   **{CM_INVALIDATED: True, CM_SUPERSEDED_BY: "other"})
    dirty = _memory("dirty", **{CM_DERIVED_DIRTY: True})

    reasons = _reasons(both, dirty)
    assert reasons[str(both.id)] == "invalidated"
    assert reasons[str(dirty.id)] == "dirty"
