"""Unit tests for import format adapters.

Inline fixtures mirror the VERIFIED export shapes (sources listed in
private orivory-private docs/superpowers/plans/2026-09-02-import-paths.md and docs/API.md §Imports):
ChatGPT mapping-DAG conversations.json, Claude chat_messages export.
CI-safe: pure parsing, no DB, no Chroma.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from app.ingestion.import_formats import (
    MAX_CONTENT_CHARS,
    SOURCE_TYPE_FOR_FORMAT,
    TRUNCATION_MARKER,
    ImportFormatError,
    ImportItem,
    _to_utc_datetime,
    detect_format,
    parse_chatgpt,
    parse_claude,
    parse_copilot,
    parse_gemini,
    parse_generic,
    parse_import,
    parse_openclaw,
)

# Real ChatGPT conversations.json shape: array of conversations; mapping is a
# DAG keyed by node id; entries may carry message: null; parts may contain
# nulls; create_time is a unix float (conversation-level) — verified against
# PAM importer-mappings.md §1.
CHATGPT_SAMPLE = [
    {
        "id": "6ef2b3a1-2c4d-4e2f-9a1b-000000000001",
        "title": "Postgres indexing advice",
        "create_time": 1738454400.0,
        "update_time": 1738454500.0,
        "mapping": {
            "root": {"id": "root", "message": None, "parent": None, "children": ["u", "a"]},
            "u": {
                "id": "u",
                "parent": "root",
                "children": ["a"],
                "message": {
                    "id": "u",
                    "author": {"role": "user"},
                    "create_time": 1738454400.5,
                    "content": {"content_type": "text", "parts": ["How should I index a jsonb column?"]},
                },
            },
            "a": {
                "id": "a",
                "parent": "u",
                "children": [],
                "message": {
                    "id": "a",
                    "author": {"role": "assistant"},
                    "create_time": 1738454410.9,
                    "content": {"content_type": "text", "parts": ["Use a GIN index on the jsonb path expression.", None]},
                    "metadata": {"model_slug": "gpt-4o"},
                },
            },
        },
    },
    {
        # system-role message only → no user knowledge → skipped
        "id": "skipped-empty",
        "title": "No user/assistant text",
        "create_time": 1738454600.0,
        "mapping": {
            "s": {
                "id": "s",
                "message": {
                    "author": {"role": "system"},
                    "create_time": 1.0,
                    "content": {"parts": ["You are ChatGPT."]},
                },
            }
        },
    },
]


def test_source_type_mapping_locked():
    """Cross-task contract: these exact strings appear in the MemoryCreate /
    list_memories Literals (Task 4) and the service's Memory rows (Task 3)."""
    assert SOURCE_TYPE_FOR_FORMAT == {
        "chatgpt": "chatgpt_import",
        "claude": "claude_import",
        "gemini": "gemini_import",
        "copilot": "copilot_import",
        "openclaw": "openclaw_import",
        "generic": "generic_import",
    }


def test_parse_chatgpt_one_memory_per_conversation():
    items = parse_chatgpt(CHATGPT_SAMPLE)
    assert len(items) == 1
    item = items[0]
    assert item.title == "Postgres indexing advice"
    assert item.source_ref == "6ef2b3a1-2c4d-4e2f-9a1b-000000000001"
    assert item.captured_at == datetime.fromtimestamp(1738454400.0, tz=UTC)
    assert "User: How should I index a jsonb column?" in item.content
    assert "Assistant: Use a GIN index on the jsonb path expression." in item.content
    # ordered by message create_time (user turn first)
    assert item.content.index("User:") < item.content.index("Assistant:")
    assert item.tags == ["chatgpt"]
    assert item.metadata == {"import_format": "chatgpt"}


def test_parse_chatgpt_skips_conversations_without_user_text():
    items = parse_chatgpt(CHATGPT_SAMPLE)
    assert [i.source_ref for i in items] == ["6ef2b3a1-2c4d-4e2f-9a1b-000000000001"]


def test_parse_chatgpt_requires_array():
    with pytest.raises(ImportFormatError, match=r"conversations\.json JSON array"):
        parse_chatgpt({"mapping": {}})


def test_to_utc_datetime_variants():
    epoch = _to_utc_datetime(1738454400.5)
    assert epoch == datetime.fromtimestamp(1738454400.5, tz=UTC)
    assert _to_utc_datetime(1738454400).tzinfo is not None  # int epochs too
    assert _to_utc_datetime(0).tzinfo is not None           # unix 0 → now(UTC) fallback
    assert _to_utc_datetime(None).tzinfo is not None       # missing → now(UTC)
    iso = _to_utc_datetime("2026-01-20T09:15:00Z")
    assert iso == datetime(2026, 1, 20, 9, 15, tzinfo=UTC)  # Z suffix handled
    naive = _to_utc_datetime("2026-01-20T09:15:00")
    assert naive.tzinfo is not None                         # naive → assumed UTC
    garbage = _to_utc_datetime("not-a-date")
    assert garbage.tzinfo is not None                       # garbage → now(UTC)


def test_import_item_rejects_empty_content():
    from pydantic import ValidationError

    # ValidationError subclasses ValueError; adapters skip empties before
    # constructing items, so this is a belt-and-suspenders guard.
    with pytest.raises(ValidationError):
        ImportItem(content="")


# ————————————————— Task 2: Claude / generic / PAM / dispatch ————————————————

# Real Claude conversations.json shape: array; chat_messages (NOT messages);
# sender "human"/"assistant"; message-level text is the plain-text duplicate
# of content[] text parts; created_at ISO 8601 — verified against PAM
# importer-mappings.md §2.
CLAUDE_SAMPLE = [
    {
        "uuid": "9f1c2ab3-1111-4222-8333-444444444444",
        "name": "Vector db selection",
        "summary": "Choosing a vector database",
        "created_at": "2026-01-20T09:15:00Z",
        "updated_at": "2026-01-20T09:20:00Z",
        "chat_messages": [
            {
                "uuid": "m1",
                "sender": "human",
                "text": "pgvector or Qdrant for 10M vectors?",
                "created_at": "2026-01-20T09:15:00Z",
                "content": [],
            },
            {
                "uuid": "m2",
                "sender": "assistant",
                "text": "pgvector keeps data in Postgres; Qdrant separates storage.",
                "created_at": "2026-01-20T09:16:00Z",
                "content": [{"type": "text", "text": "pgvector keeps data in Postgres; Qdrant separates storage."}],
            },
        ],
    },
    {"uuid": "empty-conv", "name": "No text", "chat_messages": []},
]

# Real PAM memory-store.json shape (minimal example from portable-ai-memory.org).
PAM_SAMPLE = {
    "schema": "portable-ai-memory",
    "schema_version": "1.0",
    "owner": {"id": "user-123"},
    "memories": [
        {
            "id": "mem-001",
            "type": "skill",
            "content": "User is a cloud infrastructure engineer",
            "temporal": {"created_at": "2025-01-15T00:00:00Z"},
            "provenance": {"platform": "chatgpt"},
        },
        {
            "id": "mem-002",
            "type": "preference",
            "content": "Prefers concise answers",
            "temporal": {"created_at": "2025-02-01T00:00:00Z"},
            "provenance": {"platform": "claude"},
        },
    ],
}

GENERIC_SAMPLE = [
    {
        "title": "Rewind daily digest",
        "content": "Worked on the import path today.",
        "created_at": "2026-09-01T10:00:00Z",
        "url": "https://example.com/d",
        "tags": ["rewind"],
        "ref": "rewind-2026-09-01",
    },
    {"content": "untitled note"},
    {"title": "no content — skipped"},
]


def test_parse_claude_one_memory_per_conversation():
    items = parse_claude(CLAUDE_SAMPLE)
    assert len(items) == 1
    item = items[0]
    assert item.title == "Vector db selection"
    assert item.source_ref == "9f1c2ab3-1111-4222-8333-444444444444"
    assert item.captured_at == datetime(2026, 1, 20, 9, 15, tzinfo=UTC)
    assert item.content.startswith("User: pgvector or Qdrant")
    assert "Assistant: pgvector keeps data in Postgres" in item.content
    assert item.tags == ["claude"]
    assert item.metadata == {"import_format": "claude"}


def test_parse_claude_requires_array():
    with pytest.raises(ImportFormatError, match=r"conversations\.json JSON array"):
        parse_claude({"chat_messages": []})


def test_parse_generic_array():
    items = parse_generic(GENERIC_SAMPLE)
    assert len(items) == 2  # third entry has no content → skipped
    assert items[0].source_ref == "rewind-2026-09-01"
    assert items[0].source_url == "https://example.com/d"
    assert items[0].tags == ["rewind"]
    assert items[0].captured_at == datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
    assert items[1].title is None  # untitled note


def test_parse_generic_pam_bundle():
    items = parse_generic(PAM_SAMPLE)
    assert len(items) == 2
    first = items[0]
    assert first.title == "[skill] User is a cloud infrastructure engineer"
    assert first.source_ref == "mem-001"
    assert first.tags == ["skill"]
    assert first.metadata["pam"] is True
    assert first.metadata["platform"] == "chatgpt"
    assert first.captured_at == datetime(2025, 1, 15, tzinfo=UTC)


def test_detect_format():
    assert detect_format(CHATGPT_SAMPLE) == "chatgpt"
    assert detect_format(CLAUDE_SAMPLE) == "claude"
    assert detect_format(GENERIC_SAMPLE) == "generic"
    assert detect_format(PAM_SAMPLE) == "generic"  # PAM rides the generic path
    assert detect_format([]) == "unknown"
    assert detect_format({"stray": 1}) == "unknown"


def test_parse_import_truncates_long_content():
    payload = [{"content": "x" * (MAX_CONTENT_CHARS + 500)}]
    items = parse_import(payload, "generic")
    assert len(items) == 1
    assert len(items[0].content) == MAX_CONTENT_CHARS  # exactly at the cap
    assert items[0].content.endswith(TRUNCATION_MARKER)


def test_parse_import_caps_chatgpt_transcript():
    """Controller-mandated handoff from T1 review: the 10k cap applies to the
    FINAL assembled transcript (role prefixes + joined turns included), not
    just raw message text — and the ChatGPT path must be covered too."""
    # 1200 user turns × ("User: " + 9 chars + "\n\n") ≈ 19k chars assembled
    conv = {
        "id": "big-conv",
        "title": "Big conversation",
        "create_time": 1738454400.0,
        "mapping": {
            str(i): {
                "id": str(i),
                "message": {
                    "author": {"role": "user"},
                    "create_time": float(1738454400 + i),
                    "content": {"parts": ["abcdefghi"]},
                },
            }
            for i in range(1200)
        },
    }
    items = parse_import([conv], "chatgpt")
    assert len(items) == 1
    assert len(items[0].content) == MAX_CONTENT_CHARS
    assert items[0].content.endswith(TRUNCATION_MARKER)
    # the marker is appended, not spliced into the middle: the kept text is
    # exactly the first (cap − marker) chars of the uncapped transcript
    uncapped = parse_chatgpt([conv])[0].content
    keep = MAX_CONTENT_CHARS - len(TRUNCATION_MARKER)
    assert items[0].content == uncapped[:keep] + TRUNCATION_MARKER


def test_parse_import_dispatches_by_detected_format():
    # explicit concrete format → that parser
    assert [i.source_ref for i in parse_import(CLAUDE_SAMPLE, "claude")] == [
        "9f1c2ab3-1111-4222-8333-444444444444"
    ]
    # format None → detect_format dispatch (claude sample detected as claude)
    auto = parse_import(CLAUDE_SAMPLE)
    assert [i.tags for i in auto] == [["claude"]]
    # PAM rides the generic path under auto too
    pam = parse_import(PAM_SAMPLE)
    assert len(pam) == 2
    assert pam[0].metadata["pam"] is True


def test_parse_import_invalid_json_payload():
    """Controller-mandated: JSON decode errors surface as ImportFormatError
    (a str/bytes payload is what json.loads would have raised on)."""
    with pytest.raises(ImportFormatError):
        parse_import("not json at all {", "generic")


def test_parse_import_unknown_format_rejected():
    with pytest.raises(ImportFormatError, match=r"unknown source_format: 'rewind'"):
        parse_import([{"content": "x"}], "rewind")
    # also when detection fails: no content-bearing shape at all
    with pytest.raises(ImportFormatError, match=r"could not detect"):
        parse_import({"stray": 1})


def test_pam_malformed_provenance_degrades_never_crashes():
    """Final fix wave: wrong-typed sub-fields degrade, never crash the import.

    provenance-as-str is coerced to {} (platform: None) — the memory is kept
    with its content intact. Hard per-item errors raised by the converter
    are caught by _safe_item → skipped, never a 500.
    """
    data = {
        "schema": "portable-ai-memory",
        "memories": [
            {"id": "p1", "type": "note", "content": "good memory",
             "temporal": {"created_at": "2026-01-01T00:00:00Z"}, "provenance": {"platform": "chatgpt"}},
            {"id": "p2", "type": "note", "content": "bad memory", "provenance": "not-a-dict"},
        ],
    }
    items = parse_generic(data)
    assert [i.source_ref for i in items] == ["p1", "p2"]
    assert items[1].metadata["platform"] is None


def test_generic_created_at_as_dict_degrades_to_now():
    data = [{"content": "ok", "created_at": {"nested": True}}, {"content": "fine"}]
    items = parse_generic(data)
    assert [i.content for i in items] == ["ok", "fine"]  # _to_utc_datetime falls back to now()


def test_chatgpt_payload_level_errors_stay_format_error():
    """Format-LEVEL (payload-shape) errors stay ImportFormatError; per-item
    shape errors (e.g. mapping-as-list on ONE conversation) are skipped by
    _safe_item — one bad conversation must not 500 a whole import."""
    with pytest.raises(ImportFormatError):
        parse_import({"not": "an array"}, "chatgpt")
    payload = json.dumps([{"title": "t", "mapping": ["not", "a", "dag"]}])
    items = parse_import(payload, "chatgpt")  # skipped, not raised
    assert items == []


def test_generic_tags_as_string_ignored_not_char_split():
    data = [{"content": "x", "tags": "rewind"}]
    items = parse_generic(data)
    assert items[0].tags == []


def test_giant_create_time_overflow_degrades_never_fatal():
    """10**400 as create_time must not 500 the import: OverflowError is
    caught and the turn degrades to insertion-order (create_time=0)."""
    payload = json.dumps([
        {"title": "boom", "mapping": {"a": {"message": {"author": {"role": "user"},
         "content": {"parts": ["hi"]}, "create_time": 10**400}}}},
        {"title": "fine", "mapping": {"a": {"message": {"author": {"role": "user"},
         "content": {"parts": ["hello"]}, "create_time": 1700000000}}}},
    ])
    items = parse_import(payload, "chatgpt")
    assert sorted(i.title for i in items) == ["boom", "fine"]  # both survive


def test_gemini_takesout_activities_shape():
    data = [
        {"time": "2026-07-01T10:00:00Z", "title": "Trip planning", "text": "User: plan a trip\nAssistant: here is a plan"},
        {"title": "Empty", "text": ""},
    ]
    items = parse_gemini(data)
    assert len(items) == 1
    assert items[0].title == "Trip planning"
    assert "plan a trip" in items[0].content


def test_gemini_conversations_shape():
    data = {"conversations": [{"title": "Idea", "created_date": "2026-07-02T09:00:00Z",
                               "messages": [{"role": "user", "text": "brainstorm"}]}]}
    items = parse_gemini(data)
    assert len(items) == 1 and "brainstorm" in items[0].content


def test_copilot_thread_shape():
    data = {"conversations": [{"conversation_id": "c1", "title": "Copilot chat",
                               "thread": [{"role": "user", "text": "write tests"}]}]}
    items = parse_copilot(data)
    assert len(items) == 1 and items[0].source_ref == "c1"


def test_openclaw_session_shape():
    data = {"memory_entries": [{"session_id": "s1",
                                "entries": [{"role": "user", "content": "ship it"}]}]}
    items = parse_openclaw(data)
    assert len(items) == 1 and "ship it" in items[0].content
    assert items[0].tags == ["openclaw"]


def test_detect_openclaw_and_gemini():
    assert detect_format({"memory_entries": [{"session_id": "s"}]}) == "openclaw"
    assert detect_format([{"session_id": "s", "memory_entries": []}]) == "openclaw"
    # A PAM memory store WITHOUT the schema marker: memories + export_version.
    # It used to be routed to the gemini adapter, which only reads
    # ``conversations`` — a successful import that persisted ZERO memories.
    assert detect_format({"memories": [], "export_version": "1"}) == "generic"


def test_memories_export_imports_its_memories_not_nothing():
    """id 348(a): ``{'memories': [...], 'export_version': ...}`` parsed 0 items."""
    data = {
        "export_version": "1.0",
        "memories": [
            {"id": "m1", "type": "fact", "content": "Alice ships the wave",
             "temporal": {"created_at": "2026-07-03T08:00:00Z"}},
            {"id": "m2", "type": "note", "content": "Bob reviews it"},
        ],
    }

    items = parse_import(data)  # auto-detected

    assert detect_format(data) == "generic"
    assert [item.content for item in items] == ["Alice ships the wave", "Bob reviews it"]
    assert [item.source_ref for item in items] == ["m1", "m2"]


def test_gemini_conversations_dict_is_detected_and_parsed():
    """id 348(b): the documented Gemini shape was ``unknown`` → ImportFormatError."""
    data = {"conversations": [{"title": "Idea", "created_date": "2026-07-02T09:00:00Z",
                               "messages": [{"role": "user", "text": "brainstorm"}]}]}

    assert detect_format(data) == "gemini"
    items = parse_import(data)  # the caller does not have to name the format
    assert len(items) == 1 and "brainstorm" in items[0].content


def test_gemini_lookalike_without_a_text_turn_is_not_gemini():
    """id 348 follow-up: the detection must not swallow a non-Gemini shape.

    ``{"conversations": [{"messages": [{"author":…, "content": {"parts": […]}}]}]}``
    is NOT the documented Gemini shape (``messages: [{role, text}]``) and
    ``parse_gemini`` reads only ``text``: detecting it made ``_safe_item``
    swallow every conversation and report a SUCCESSFUL import of ZERO memories,
    where the unknown path fails loudly.
    """
    data = {"conversations": [{"title": "Idea", "messages": [
        {"author": "user", "content": {"parts": ["brainstorm"]}},
    ]}]}

    assert detect_format(data) == "unknown"
    with pytest.raises(ImportFormatError, match="could not detect"):
        parse_import(data)
