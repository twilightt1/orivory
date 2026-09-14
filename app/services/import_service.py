"""One-shot memory import service (Open Memory Hub MVP item 3).

Turns an uploaded provider export file into Memory rows: parse (adapters)
→ resolve/detect format → dedup on (user_id, source_type, source_ref) →
create rows → single commit → best-effort indexing (embed + graph enqueue
via write_back — Postgres is the source of truth, indexing never raises).

v0 is synchronous and bounded by the router's 20 MiB upload cap; large
imports move to a Celery task later (docs/API.md §Imports, follow-ups).

Signature/counts follow the controller's Task 3 binding rules — parse_import
is called as parse_import(data, source_format) (data first, per its shipped
signature), and the summary counts are {parsed, created, skipped_duplicates,
failed, index_failures}.
"""
from __future__ import annotations

import json
import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.import_formats import (
    SOURCE_TYPE_FOR_FORMAT,
    ImportFormatError,
    detect_format,
    parse_import,
)
from app.models.memory import Memory
from app.retrieval.embedder import EmbeddingDimensionMismatch
from app.retrieval.memory.outbox import bump_revision, enqueue_upsert
from app.retrieval.memory.write_back import index_new_memory
from app.schemas.Orivory import ImportSummary

log = logging.getLogger(__name__)


async def run_import(
    db: AsyncSession,
    user_id: uuid.UUID,
    raw_data: bytes,
    source_format: str | None,
    *,
    requested_by: str,
) -> ImportSummary:
    """Import one export file's raw bytes for one user.

    Raises ``ImportFormatError`` for undecodable JSON or an undetectable
    format; per-item problems are isolated into the ``failed`` counter
    instead of failing the whole run (house pattern: SourceSyncService).
    Indexing failures are counted for ordinary transient outages; a typed
    embedding contract failure is raised so the caller cannot miss a data
    integrity/readiness blocker.
    """
    try:
        parsed = json.loads(raw_data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ImportFormatError(f"import payload is not valid UTF-8 JSON: {exc}") from exc

    detected = source_format if source_format not in (None, "auto") else detect_format(parsed)
    if detected not in SOURCE_TYPE_FOR_FORMAT:
        raise ImportFormatError(
            "could not detect the export format — pass source_format "
            "explicitly (chatgpt | claude | generic)"
        )
    source_type = SOURCE_TYPE_FOR_FORMAT[detected]

    # parse_import(data, source_format=None) — data first (shipped T1-T2
    # signature). JSON is already decoded here, so the str/bytes decode
    # branch inside parse_import stays a defensive no-op for this caller.
    items = parse_import(parsed, detected)

    # Dedup part 1: refs this user already imported from this format.
    # One batched select (never N per-item queries); memories has no unique
    # index, so the query is the dedup authority — scoped to user + source_type.
    refs = [item.source_ref for item in items if item.source_ref]
    existing_refs: set[str] = set()
    if refs:
        rows = (
            await db.execute(
                select(Memory.source_ref).where(
                    Memory.user_id == user_id,
                    Memory.source_type == source_type,
                    Memory.source_ref.in_(refs),
                )
            )
        ).scalars().all()
        existing_refs = {row for row in rows if row}

    created_rows: list[Memory] = []
    seen_refs: set[str] = set()
    skipped_duplicates = 0
    failed = 0
    index_failures = 0

    # Dedup part 2: within the same file.
    for item in items:
        try:
            if item.source_ref and (item.source_ref in existing_refs or item.source_ref in seen_refs):
                skipped_duplicates += 1
                continue
            memory = Memory(
                user_id=user_id,
                title=item.title,
                content=item.content,  # already capped by parse_import
                source_type=source_type,
                source_ref=item.source_ref,
                source_url=item.source_url,
                tags=item.tags,
                captured_at=item.captured_at,
                extra_metadata={**item.metadata, "import": {"requested_by": requested_by}},
            )
            # Stamp the first indexed revision at construction: a later flush
            # would materialize the column default first and turn the same
            # bump into revision 2.
            bump_revision(memory)
            db.add(memory)
            created_rows.append(memory)
            if item.source_ref:
                seen_refs.add(item.source_ref)
        except Exception as exc:  # per-item isolation; the run continues
            failed += 1
            log.warning(
                "Import item failed",
                extra={"user_id": str(user_id), "source_type": source_type, "error": str(exc)},
            )

    if created_rows:
        # Durable index intents for the whole batch, in the same commit as the
        # rows themselves — a crash before indexing replays from the outbox.
        for memory in created_rows:
            await enqueue_upsert(db, memory)
        await db.commit()
        # Server-side defaults (id, timestamps) must load before indexing.
        for memory in created_rows:
            await db.refresh(memory)
        # Best-effort embed + graph enqueue — a failure is counted, not raised;
        # the committed Postgres row is the truth (reindex task replays it).
        for memory in created_rows:
            try:
                await index_new_memory(memory)
            except EmbeddingDimensionMismatch:
                raise
            except Exception as exc:
                index_failures += 1
                log.warning(
                    "Import index_new_memory failed",
                    extra={"user_id": str(user_id), "error": str(exc)},
                )

    return ImportSummary(
        parsed=len(items),
        created=len(created_rows),
        skipped_duplicates=skipped_duplicates,
        failed=failed,
        index_failures=index_failures,
    )


__all__ = ["run_import"]
