"""
SourceSyncService — the dispatcher that turns connector output into Memory rows.

For one Source row:
    1. Pick the right connector from the registry.
    2. Validate config.
    3. Call `connector.fetch_items()`.
    4. For each item, check if a memory with the same
       `(source_id, source_ref)` already exists:
         - skip if it does (idempotent re-sync)
         - update if the upstream content changed
         - create otherwise
    5. Update Source's `last_sync_at`, `memories_synced`, and `status`.

Returns a `SyncResult` with counts and per-item errors.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.ingestion.connectors.registry import get_connector_for_source
from app.ingestion.document_memory import is_suppressed_async
from app.ingestion.types import ConnectorItem, ItemError, SyncResult
from app.models.memory import Memory
from app.models.source import MemorySource, Source
from app.retrieval.memory.outbox import bump_revision, enqueue_upsert, mark_done

log = logging.getLogger(__name__)


class SourceSyncService:
    """Coordinates one `Source.sync()` invocation."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def sync(self, source: Source) -> SyncResult:
        started_at = datetime.now(UTC)
        result = SyncResult(
            source_id=str(source.id),
            started_at=started_at,
            finished_at=started_at,  # updated at the end
        )
        # Memories whose content was created or changed this sync — they need
        # (re)indexing (embed + graph) after the transaction commits.
        index_memory_ids: list[str] = []

        # 1. Pick the connector (Phase 2.6: pass initial_cursor for incremental sync)
        try:
            connector = get_connector_for_source(
                source.source_type, source.config or {},
                initial_cursor=source.sync_cursor,
            )
        except KeyError as e:
            result.errors.append(ItemError(message=str(e)))
            result.notes.append("No connector registered for this source_type.")
            return await self._finalize(source, result)

        # 2. Validate
        try:
            connector.validate_config()
        except ValueError as e:
            result.errors.append(ItemError(message=f"Config invalid: {e}"))
            result.notes.append("Source config is missing required fields.")
            return await self._finalize(source, result)

        # 3. Fetch
        try:
            items = await connector.fetch_items()
        except NotImplementedError as e:
            result.errors.append(ItemError(message=str(e) or "Connector not yet implemented."))
            result.notes.append("This connector is a stub in Phase 2 v0.")
            return await self._finalize(source, result)
        except Exception as e:
            log.exception("Connector fetch failed for source %s", source.id)
            result.errors.append(ItemError(message=f"Fetch failed: {e}"))
            return await self._finalize(source, result)

        result.items_yielded = len(items)

        # Surface any per-item fetch failures the connector collected (failed
        # URLs, unparseable feed entries) without failing the whole sync.
        if getattr(connector, "fetch_errors", None):
            result.errors.extend(connector.fetch_errors)

        # Phase 2.6: save the last pagination token so the next sync
        # resumes here instead of re-fetching everything.
        # Only update if the connector set one (None means exhausted
        # or not applicable — keep existing cursor in that case).
        if connector.last_cursor is not None:
            source.sync_cursor = connector.last_cursor

        # 4. Persist each item, deduping by (source_id, source_ref)
        for item in items:
            try:
                outcome, memory_id = await self._persist_item(source, item)
                if outcome == "added":
                    result.memories_added += 1
                    if memory_id:
                        index_memory_ids.append(memory_id)
                elif outcome == "updated":
                    result.memories_updated += 1
                    # Content changed → vector + graph are now stale, reindex.
                    if memory_id:
                        index_memory_ids.append(memory_id)
                elif outcome == "skipped":
                    result.memories_skipped += 1
            except Exception as e:
                log.exception("Persist failed for item %r", item.source_ref)
                result.errors.append(ItemError(source_ref=item.source_ref, message=str(e)))

        finalized = await self._finalize(source, result)
        commit_failed = any((err.message or "").startswith("Commit failed") for err in finalized.errors)
        if index_memory_ids and not commit_failed:
            await self._index_memories(index_memory_ids, user_id=source.user_id)
        return finalized

    async def _index_memories(self, memory_ids: list[str], *, user_id) -> None:
        """Embed + enqueue graph extraction for committed memories.

        Best-effort: the memories are already durable in Postgres (with their
        index intents), so an embedding/enqueue failure is logged and replayable
        via the reindex task rather than failing the sync. Scoped to the source
        owner so a foreign id can never be embedded through this sync.
        """
        from app.retrieval.memory.write_back import (
            safe_enqueue_graph_build,
            safe_upsert_to_index,
        )

        rows = (
            await self.db.execute(
                select(Memory).where(
                    Memory.id.in_(memory_ids),
                    Memory.user_id == user_id,
                )
            )
        ).scalars().all()
        for memory in rows:
            if await safe_upsert_to_index(memory):
                # Indexed now: ack the intent this batch committed, so a boot
                # drain does not re-embed it.
                await mark_done(self.db, entity_id=memory.id, revision=memory.revision)
            # Scheduled, never awaited (R27(p2)): on this loop the sync helper
            # hands the build to a background task (a cut-off build is an
            # accepted best-effort loss; failures log at ERROR) and never raises.
            safe_enqueue_graph_build(memory.id)

    # ── internals ────────────────────────────────────────────────────────────

    async def _persist_item(self, source: Source, item: ConnectorItem) -> tuple[str, str | None]:
        """
        Create or update a Memory + MemorySource pair for one item.
        Returns (outcome, memory_id), where outcome is 'added' | 'updated' | 'skipped'.

        A suppressed identity is skipped outright: re-sync must never resurrect
        a memory the user forgot (spec §5.4). Every created/updated row gets a
        durable index intent in the batch transaction (spec §5.1).
        """
        if item.source_ref and await is_suppressed_async(
            self.db, user_id=source.user_id, source_ref=item.source_ref
        ):
            return "skipped", None

        # Find an existing memory linked to this (source, ref).
        # Eager-load the memory relationship — lazy loading on AsyncSession
        # raises MissingGreenlet.
        existing_link = await self.db.scalar(
            select(MemorySource)
            .options(selectinload(MemorySource.memory))
            .where(
                MemorySource.source_id == source.id,
                MemorySource.item_ref == item.source_ref,
            )
        )

        if existing_link and existing_link.memory:
            # Idempotency: same source_ref => skip unless content changed
            memory = existing_link.memory
            if (memory.title == item.title and
                memory.content == item.content and
                memory.summary == item.summary):
                return "skipped", str(memory.id)
            # Update
            memory.title = item.title
            memory.content = item.content
            memory.summary = item.summary
            memory.tags = item.tags
            memory.extra_metadata = {**(memory.extra_metadata or {}), **item.metadata}
            existing_link.item_excerpt = item.source_excerpt
            existing_link.item_url = item.source_url
            bump_revision(memory)
            await enqueue_upsert(self.db, memory)
            return "updated", str(memory.id)

        # Create new memory
        memory = Memory(
            user_id=source.user_id,
            title=item.title,
            content=item.content,
            summary=item.summary,
            source_type=_memory_source_type_for(source.source_type),
            source_ref=item.source_ref,
            source_url=item.source_url,
            tags=item.tags,
            captured_at=item.captured_at,
            extra_metadata=item.metadata,
        )
        self.db.add(memory)
        bump_revision(memory)  # revision 1, explicit before the INSERT
        await db_flush(self.db)
        await enqueue_upsert(self.db, memory)

        link = MemorySource(
            memory_id=memory.id,
            source_id=source.id,
            item_ref=item.source_ref,
            item_url=item.source_url,
            item_excerpt=item.source_excerpt,
        )
        self.db.add(link)
        # Flush so the next item's (source, ref) lookup sees this row: the
        # batch runs on an autoflush=False session and a repeated ref in one
        # fetch must update this row, not create a duplicate.
        await db_flush(self.db)
        return "added", str(memory.id)

    async def _finalize(self, source: Source, result: SyncResult) -> SyncResult:
        """Update the Source row to reflect the sync result."""
        source.last_sync_at = result.finished_at
        source.memories_synced = (source.memories_synced or 0) + result.memories_added
        first_err = result.errors[0].message if result.errors else None
        source.sync_error = first_err
        if result.errors:
            source.status = "error"
            # Phase 2.6: also store last error inside the JSONB config
            # so the admin UI / future webhook can inspect it without
            # a separate column. Reassign (not in-place) so SQLAlchemy
            # always sees the change.
            source.config = {
                **(source.config or {}),
                "last_error":    first_err,
                "last_error_at": result.finished_at.isoformat(),
            }
        else:
            source.status = "connected"
            # Clear any previous error from config
            cfg = dict(source.config or {})
            cfg.pop("last_error", None)
            cfg.pop("last_error_at", None)
            source.config = cfg
        result.finished_at = datetime.now(UTC)
        try:
            await self.db.commit()
        except Exception as e:
            log.exception("Failed to commit sync result for source %s", source.id)
            await self.db.rollback()
            result.errors.append(ItemError(message=f"Commit failed: {e}"))
        return result


# ── helpers ────────────────────────────────────────────────────────────────

async def db_flush(db: AsyncSession) -> None:
    """Flush pending writes so we can read server-generated values (id, etc.)."""
    await db.flush()


def _memory_source_type_for(connector_source_type: str) -> str:
    """Map a Source.source_type to a Memory.source_type."""
    mapping = {
        "manual":      "manual_note",
        "file_upload": "file_upload",
        "web_clipper": "web_clipper",
        "rss":         "rss",
        "google_drive": "google_drive",
        "notion":      "notion",
        "gmail":       "gmail",
    }
    return mapping.get(connector_source_type, "other")
