"""Entity extraction: the deterministic fallback and the LLM path's schema gate.

The LLM client is stubbed (``tests/graph/conftest.py``): no provider, no
network. The fallback assertions run the real regexes against a real
Memory-shaped object.
"""
from __future__ import annotations

import uuid

from app.graph.extraction import _CAPITALIZED_PHRASE_RE, _fallback_entities, extract_entities
from app.models.memory import Memory
from tests.graph.conftest import stub_llm

# The finding's probe: the fallback's input is the ``\n``-joined
# title/summary/tags/content, so a separator that matches newlines fuses the
# last capitalized word of one line with the first of the next.
# A detached ORM instance: extraction only reads the memory's text fields.
FUSING_MEMORY = Memory(
    user_id=uuid.uuid4(),
    title="Note about Alice",
    summary="Python async",
    tags=["tagged"],
    content="Done with Bob\nSQLite tips",
)


def test_the_capitalized_phrase_fallback_never_crosses_a_line():
    """id 359: ``\\s+`` matched the newline, storing 'Alice Python'/'Bob SQLite'."""
    names = [entity.name for entity in _fallback_entities(FUSING_MEMORY)]

    assert "Alice Python" not in names, f"fused across lines: {names}"
    assert "Bob SQLite" not in names, f"fused across lines: {names}"
    assert {"Alice", "Python", "Bob", "SQLite"} <= set(names)


def test_the_capitalized_phrase_regex_stays_inside_one_line():
    """The regex face of the same fix, on the finding's literal input."""
    matches = _CAPITALIZED_PHRASE_RE.findall("Note about Alice\nPython async\nDone with Bob\nSQLite tips\n")

    assert matches == ["Note", "Alice", "Python", "Done", "Bob", "SQLite"]


async def test_a_non_list_entities_payload_is_a_schema_error(monkeypatch):
    """id 363: ``{"entities": {...}}`` was coerced to ``[]`` and returned as a
    SUCCESS (``fallback_used=False, error=None``) — the caller then stamped the
    memory as extracted and nothing ever rebuilt it."""
    stub_llm(monkeypatch, ['{"entities": {"name": "Alice", "type": "person"}}'])

    result = await extract_entities(FUSING_MEMORY)

    assert result.error, "a wrong-typed payload is a parse failure, not a success"
    assert result.fallback_used is True
    assert result.entities, "the deterministic fallback still fills the graph"
