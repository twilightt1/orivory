"""One-shot import format adapters (Open Memory Hub MVP item 3).

Turn provider data-export files into normalized ``ImportItem`` rows that
``app.services.import_service`` persists as ``Memory`` rows. Every format
claim here is verified against the PAM importer mappings (Feb 2026) —
provider export formats change without notice, so each adapter is defensive:
malformed entries are skipped, never fatal.

Verified format sources (also published in docs/API.md §Imports):
    - ChatGPT conversations.json — mapping DAG, unix-float create_time,
      message.author.role, content.parts[] with nulls:
      https://github.com/portable-ai-memory/portable-ai-memory/blob/master/importer-mappings.md
    - Claude conversations.json — chat_messages (not messages), sender
      "human"/"assistant", ISO 8601, message-level text: same document §2.
    - PAM memory-store.json — https://portable-ai-memory.org/spec/v1.0/
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

#: source_format values accepted by the API (exact strings).
SOURCE_FORMATS = ("auto", "chatgpt", "claude", "gemini", "copilot", "openclaw", "generic")

#: Memory.source_type written per format (exact strings — also in the
#: MemoryCreate and list_memories Literals; no migration needed because
#: memories.source_type is an unconstrained String(32)).
SOURCE_TYPE_FOR_FORMAT = {
    "chatgpt": "chatgpt_import",
    "claude": "claude_import",
    "gemini": "gemini_import",
    "copilot": "copilot_import",
    "openclaw": "openclaw_import",
    "generic": "generic_import",
}

#: Per-memory content cap. Matches the ChatRequest.query cap (10_000) so an
#: imported conversation stays the same order of magnitude as one chat.
MAX_CONTENT_CHARS = 10_000
TRUNCATION_MARKER = "\n\n[import truncated]"


class ImportFormatError(ValueError):
    """The uploaded file does not match the requested/detected format."""


class ImportItem(BaseModel):
    """One normalized item extracted from an export file.

    Adapters skip empty items instead of constructing them — ``content``
    stays min_length=1 as a belt-and-suspenders guard.
    """

    title:       str | None      = Field(default=None, max_length=500)
    content:     str             = Field(min_length=1)
    source_ref:  str | None      = Field(default=None, max_length=500)
    source_url:  str | None      = Field(default=None, max_length=1000)
    captured_at: datetime        = Field(default_factory=lambda: datetime.now(UTC))
    tags:        list[str]       = Field(default_factory=list, max_length=50)
    metadata:    dict[str, Any]  = Field(default_factory=dict)


def _clip(value: Any, limit: int) -> str | None:
    """Coerce to a stripped str clipped to ``limit``; None for empty."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:limit]


def _to_utc_datetime(value: Any) -> datetime:
    """Epoch seconds (int/float) or ISO 8601 str → aware UTC datetime.

    Unusable values (None, 0, garbage) fall back to ``now(UTC)`` so a bad
    upstream timestamp never drops a memory. Handles the ChatGPT unix-float
    and Claude/generic ISO-8601 forms verified in the PAM mappings.
    """
    try:
        if isinstance(value, (int, float)):
            epoch = float(value)
            if epoch > 0:
                return datetime.fromtimestamp(epoch, tz=UTC)
        elif isinstance(value, str) and value.strip():
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except (ValueError, OverflowError, OSError):
        pass
    return datetime.now(UTC)


def _safe_item[ItemType](convert: Callable[[], ItemType]) -> ItemType | None:
    """Run one item's conversion; swallow per-item shape errors → None.

    Malformed-but-JSON entries (e.g. PAM ``provenance`` as a string,
    ChatGPT ``mapping`` as a list) are skipped, never fatal — a single bad
    entry must not 500 a whole import. Format-LEVEL errors (non-array
    payload, missing schema marker) stay :class:`ImportFormatError`.
    """
    try:
        return convert()
    except (AttributeError, TypeError, KeyError, ValueError, OverflowError):
        log.warning("skipping malformed import entry", exc_info=True)
        return None


def _chatgpt_transcript(conversation: dict) -> str:
    """Render a conversation's user/assistant text, ordered by create_time.

    ``mapping`` is a DAG keyed by node id (not a flat list); nodes may carry
    ``message: null`` and ``parts`` may contain nulls (verified upstream).
    system/tool roles are dropped — they carry model instructions, not
    user knowledge. Message create_time can be 0/null → falls back to 0
    so ordering degrades to insertion order, never crashes.
    """
    turns: list[tuple[float, str, str]] = []
    for node in (conversation.get("mapping") or {}).values():
        message = node.get("message") if isinstance(node, dict) else None
        if not isinstance(message, dict):
            continue
        role = (message.get("author") or {}).get("role")
        if role not in ("user", "assistant"):
            continue
        parts = (message.get("content") or {}).get("parts") or []
        text = "\n".join(p.strip() for p in parts if isinstance(p, str) and p.strip())
        if not text:
            continue
        try:
            create_time = float(message.get("create_time") or 0)
        except (TypeError, ValueError, OverflowError):
            create_time = 0.0
        turns.append((create_time, role, text))
    turns.sort(key=lambda turn: turn[0])
    return "\n\n".join(
        f"{'User' if role == 'user' else 'Assistant'}: {text}" for _ts, role, text in turns
    )


def parse_chatgpt(data: Any) -> list[ImportItem]:
    """ChatGPT conversations.json (JSON array) → one ImportItem per conversation."""
    if not isinstance(data, list):
        raise ImportFormatError(
            "ChatGPT export must be the conversations.json JSON array "
            "(Settings → Data controls → Export data)."
        )
    items: list[ImportItem] = []
    for conversation in data:
        if not isinstance(conversation, dict):
            continue
        item = _safe_item(lambda conv=conversation: _chatgpt_item(conv))
        if item is not None:
            items.append(item)
    return items


def _chatgpt_item(conversation: dict) -> ImportItem:
    content = _chatgpt_transcript(conversation)
    if not content:
        raise ValueError("empty conversation transcript")
    return ImportItem(
        title=_clip(conversation.get("title"), 500) or "Untitled ChatGPT conversation",
        content=content,
        source_ref=_clip(conversation.get("id"), 500),
        captured_at=_to_utc_datetime(conversation.get("create_time")),
        tags=["chatgpt"],
        metadata={"import_format": "chatgpt"},
    )


def parse_claude(data: Any) -> list[ImportItem]:
    """Claude conversations.json (JSON array) → one ImportItem per conversation.

    Verified shape (PAM importer-mappings §2): messages live in
    ``chat_messages`` (NOT ``messages``) as a flat linear array; senders are
    ``"human"``/``"assistant"``; the message-level ``text`` is the plain-text
    duplicate of the content[] text parts. thinking/tool_use/tool_result
    blocks are model plumbing, not user knowledge — dropped in v0.
    """
    if not isinstance(data, list):
        raise ImportFormatError(
            "Claude export must be the conversations.json JSON array "
            "(claude.ai Settings → Privacy → Export data)."
        )
    items: list[ImportItem] = []
    for conversation in data:
        if not isinstance(conversation, dict):
            continue
        item = _safe_item(lambda conv=conversation: _claude_item(conv))
        if item is not None:
            items.append(item)
    return items


def _claude_item(conversation: dict) -> ImportItem:
    lines: list[str] = []
    for message in conversation.get("chat_messages") or []:
        if not isinstance(message, dict):
            continue
        sender = message.get("sender")
        text = str(message.get("text") or "").strip()
        if not text or sender not in ("human", "assistant"):
            continue
        lines.append(f"{'User' if sender == 'human' else 'Assistant'}: {text}")
    if not lines:
        raise ValueError("empty claude conversation")
    return ImportItem(
        title=_clip(conversation.get("name"), 500) or "Untitled Claude conversation",
        content="\n\n".join(lines),
        source_ref=_clip(conversation.get("uuid"), 500),
        captured_at=_to_utc_datetime(conversation.get("created_at")),
        tags=["claude"],
        metadata={"import_format": "claude"},
    )


def _parse_pam(data: dict) -> list[ImportItem]:
    """PAM memory-store.json → one ImportItem per memory (generic path).

    PAM memories carry no title (content is the payload), so synthesize
    ``[<type>] <content preview>``. Relations/conversations companions are
    out of scope for v0.
    """
    items: list[ImportItem] = []
    for memory in data.get("memories") or []:
        if not isinstance(memory, dict):
            continue
        item = _safe_item(lambda mem=memory: _pam_item(mem))
        if item is not None:
            items.append(item)
    return items


def _pam_item(memory: dict) -> ImportItem:
    content = str(memory.get("content") or "").strip()
    if not content:
        raise ValueError("empty PAM memory content")
    mem_type = str(memory.get("type") or "memory")
    provenance = memory.get("provenance") or {}
    if not isinstance(provenance, dict):
        provenance = {}
    temporal = memory.get("temporal") or {}
    if not isinstance(temporal, dict):
        temporal = {}
    return ImportItem(
        title=_clip(f"[{mem_type}] {content[:120]}", 500),
        content=content,
        source_ref=_clip(memory.get("id"), 500),
        captured_at=_to_utc_datetime(temporal.get("created_at")),
        tags=[mem_type],
        metadata={
            "import_format": "generic",
            "pam": True,
            "platform": provenance.get("platform"),
        },
    )


def parse_gemini(data: Any) -> list[ImportItem]:
    """Gemini Apps Takeout / export → one ImportItem per conversation.

    PAM §3 documents two Takeout variants:
      - ``MyActivity.json``: array of {time, title?, text?} entries
      - ``Gemini Conversations.json``: {conversations: [{title, created_date,
        messages: [{role, text}]}]}

    Both reduce to the conversation-transcript shape. Undetectable garbage
    raises :class:`ImportFormatError`.
    """
    if isinstance(data, dict):
        convs = data.get("conversations") or []
        if not isinstance(convs, list):
            raise ImportFormatError(
                "Gemini export 'conversations' must be a JSON array."
            )
        data = convs
    if not isinstance(data, list):
        raise ImportFormatError(
            "Gemini export must be a JSON array of activities or "
            "{'conversations': [...]} object."
        )
    items: list[ImportItem] = []
    for conv in data:
        item = _safe_item(lambda c=conv: _gemini_item(c))
        if item is not None:
            items.append(item)
    return items


def _gemini_item(conv: dict) -> ImportItem:
    title = _clip(conv.get("title") or conv.get("text"), 500) or "Untitled Gemini conversation"
    turns = conv.get("messages") or []
    if not isinstance(turns, list):
        turns = []
    lines: list[str] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "user")
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        label = "User" if role.lower() in ("user", "human") else "Assistant"
        lines.append(f"{label}: {text}")
    content = "\n\n".join(lines) if lines else str(conv.get("text") or "").strip()
    if not content:
        raise ValueError("empty gemini conversation")
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS] + TRUNCATION_MARKER
    created = _to_utc_datetime(conv.get("created_date") or conv.get("time"))
    return ImportItem(
        title=title,
        content=content,
        source_ref=_clip(conv.get("id") or conv.get("title"), 500),
        captured_at=created,
        tags=["gemini"],
        metadata={"import_format": "gemini"},
    )


def parse_copilot(data: Any) -> list[ImportItem]:
    """GitHub Copilot export (PAM §4) → one ImportItem per conversation.

    PAM §4 documents the Privacy-Dashboard export: a JSON array of
    ``{conversation_id, thread: {messages: [{role, text?}]}, ...}``-ish
    shapes. Fields are defensive: everything optional, empty entries skip.
    """
    convs = data.get("conversations") if isinstance(data, dict) else data
    if not isinstance(convs, list):
        raise ImportFormatError(
            "Copilot export must be a JSON array of conversations "
            "or {'conversations': [...]}."
        )
    items: list[ImportItem] = []
    for conv in convs:
        item = _safe_item(lambda c=conv: _copilot_item(c))
        if item is not None:
            items.append(item)
    return items


def _copilot_item(conv: dict) -> ImportItem:
    thread = conv.get("thread") or conv.get("messages") or []
    if not isinstance(thread, list):
        raise ValueError("copilot thread is not a list")
    lines: list[str] = []
    for turn in thread:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "user")
        text = str(turn.get("text") or turn.get("content") or "").strip()
        if not text:
            continue
        label = "User" if role.lower() in ("user", "human") else "Assistant"
        lines.append(f"{label}: {text}")
    if not lines:
        raise ValueError("empty copilot conversation")
    content = "\n\n".join(lines)
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS] + TRUNCATION_MARKER
    title = _clip(conv.get("title"), 500) or "Untitled Copilot conversation"
    return ImportItem(
        title=title,
        content=content,
        source_ref=_clip(conv.get("conversation_id") or conv.get("id"), 500),
        captured_at=_to_utc_datetime(conv.get("created_at")),
        tags=["copilot"],
        metadata={"import_format": "copilot"},
    )


def parse_openclaw(data: Any) -> list[ImportItem]:
    """OpenClaw session-log dump → one ImportItem per session.

    OpenClaw stores agent memory as local Markdown state (see
    private orivory-private docs/research/PLATFORM_LANDSCAPE.md §4). The converter accepts the
    JSON shape produced by the documented export flow: a list of
    ``{session_id, entries: [{role, content, timestamp?}]}`` records —
    i.e. the natural dump of the session-log store.
    """
    sessions = data.get("memory_entries") if isinstance(data, dict) else data
    if sessions is None and isinstance(data, dict) and "session_id" in data:
        sessions = [data]  # a single session record is also valid
    if not isinstance(sessions, list):
        raise ImportFormatError(
            "OpenClaw export must be a JSON array of session records, "
            "a single session record, or {'memory_entries': [...]}."
        )
    items: list[ImportItem] = []
    for session in sessions:
        item = _safe_item(lambda s=session: _openclaw_item(s))
        if item is not None:
            items.append(item)
    return items


def _openclaw_item(session: dict) -> ImportItem:
    session_id = str(session.get("session_id") or "").strip()
    entries = session.get("entries") or session.get("logs") or []
    if not isinstance(entries, list):
        raise ValueError("openclaw session entries are not a list")
    lines: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            entry = {"role": "user", "content": entry}
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role") or "user")
        text = str(entry.get("content") or entry.get("text") or "").strip()
        if not text:
            continue
        label = "User" if role.lower() in ("user", "human") else "Assistant"
        lines.append(f"{label}: {text}")
    if not lines:
        raise ValueError("empty openclaw session")
    content = "\n\n".join(lines)
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS] + TRUNCATION_MARKER
    title = _clip(session.get("title"), 500) or f"OpenClaw session {session_id[:8]}" if session_id else "OpenClaw session"
    return ImportItem(
        title=title,
        content=content,
        source_ref=_clip(session_id or session.get("id"), 500) or None,
        captured_at=_to_utc_datetime(session.get("created_at") or session.get("timestamp")),
        tags=["openclaw"],
        metadata={"import_format": "openclaw"},
    )


def parse_generic(data: Any) -> list[ImportItem]:
    """Generic JSON → ImportItems.

    Accepts either a PAM ``memory-store.json`` (dict with
    ``schema == "portable-ai-memory"``) or a JSON array of
    ``{title?, content, created_at?, url?, ref?, tags?}`` items — the shape
    any provider (or an OpenRecall sqlite dump, see docs) can be reduced to.
    """
    if isinstance(data, dict) and data.get("schema") == "portable-ai-memory":
        return _parse_pam(data)
    if not isinstance(data, list):
        raise ImportFormatError(
            "Generic import must be a JSON array of "
            "{title?, content, created_at?, url?, ref?, tags?} items "
            "or a PAM memory-store.json object."
        )
    items: list[ImportItem] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        item = _safe_item(lambda e=entry: _generic_item(e))
        if item is not None:
            items.append(item)
    return items


def _generic_item(entry: dict) -> ImportItem:
    content = str(entry.get("content") or "").strip()
    if not content:
        raise ValueError("empty generic item content")
    raw_tags = entry.get("tags")
    # Non-list tags (e.g. a bare string) have ambiguous semantics — ignore
    # them rather than char-splitting a string into fake per-character tags.
    tags = [str(t) for t in raw_tags if str(t).strip()][:50] if isinstance(raw_tags, list) else []
    return ImportItem(
        title=_clip(entry.get("title"), 500),
        content=content,
        source_ref=_clip(entry.get("ref"), 500),
        source_url=_clip(entry.get("url"), 1000),
        captured_at=_to_utc_datetime(entry.get("created_at")),
        tags=tags,
        metadata={"import_format": "generic"},
    )


def detect_format(data: Any) -> str:
    """Best-effort provider detection (PAM spec §9 heuristics, verified).

    Keys off the first array element: ``mapping`` → chatgpt,
    ``chat_messages`` → claude, ``content`` → generic; a dict with the PAM
    ``schema`` marker → generic.
    """
    if isinstance(data, list):
        sample = data[0] if data else {}
        if isinstance(sample, dict):
            if "mapping" in sample:
                return "chatgpt"
            if "chat_messages" in sample:
                return "claude"
            if "session_id" in sample and "memory_entries" in sample:
                return "openclaw"
            if "content" in sample:
                return "generic"
    if isinstance(data, dict):
        if data.get("schema") == "portable-ai-memory":
            return "generic"
        if "session_id" in data and isinstance(data.get("entries"), list):
            return "openclaw"
        if "memories" in data and "export_version" in data:
            return "gemini"
        if "memory_entries" in data:
            return "openclaw"
        if "conversations" in data and "copilot" in str(data.keys()).lower():
            return "copilot"
        if "memory_entries" in data:
            return "openclaw"
    return "unknown"


_PARSERS: dict[str, Any] = {
    "chatgpt": parse_chatgpt,
    "claude": parse_claude,
    "gemini": parse_gemini,
    "copilot": parse_copilot,
    "openclaw": parse_openclaw,
    "generic": parse_generic,
}


def _cap_content(items: list[ImportItem]) -> list[ImportItem]:
    """Clip every item's content to exactly ``MAX_CONTENT_CHARS``.

    The cap applies to the FINAL assembled content — role prefixes and
    joined turns included, not just raw message text. T1 review handoff.
    """
    for item in items:
        if len(item.content) > MAX_CONTENT_CHARS:
            keep = MAX_CONTENT_CHARS - len(TRUNCATION_MARKER)
            item.content = item.content[:keep] + TRUNCATION_MARKER
    return items


def parse_import(data: Any, source_format: str | None = None) -> list[ImportItem]:
    """Parse an export payload → ImportItems with content capped at 10k.

    ``source_format`` may be a concrete format (``"chatgpt" | "claude" |
    "generic"``) or None to auto-detect via :func:`detect_format` (the
    service layer passes the user's explicit choice through; ``"auto"``
    resolves to None here). Payloads that are raw JSON text (str/bytes —
    what a caller that has not decoded yet hands us) are decoded here,
    with decode errors surfaced as ``ImportFormatError``.
    """
    if isinstance(data, (str, bytes, bytearray)):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ImportFormatError(f"import payload is not valid JSON: {exc}") from exc
    if source_format in (None, "auto"):
        source_format = detect_format(data)
        if source_format == "unknown":
            raise ImportFormatError(
                "could not detect the export format — pass source_format "
                "explicitly (chatgpt | claude | generic)."
            )
    parser = _PARSERS.get(source_format)
    if parser is None:
        raise ImportFormatError(
            f"unknown source_format: {source_format!r} (supported: {SOURCE_FORMATS})"
        )
    return _cap_content(parser(data))


__all__ = [
    "MAX_CONTENT_CHARS",
    "SOURCE_FORMATS",
    "SOURCE_TYPE_FOR_FORMAT",
    "TRUNCATION_MARKER",
    "ImportFormatError",
    "ImportItem",
    "detect_format",
    "parse_chatgpt",
    "parse_claude",
    "parse_generic",
    "parse_import",
]
