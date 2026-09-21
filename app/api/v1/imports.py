"""
Imports API — one-shot provider export imports (Open Memory Hub MVP 3).

Endpoint:
    POST /api/v1/imports    multipart upload: file (export JSON) + optional
                            source_format form field → 201 ImportSummary
                            {parsed, created, skipped_duplicates, failed,
                            index_failures}

v0 is synchronous and capped at 20 MiB per upload; larger imports move to a
Celery task in a later plan (docs/API.md §Imports). Format claims and
export paths are verified against the PAM importer mappings — see
docs/API.md §Imports for the source list.

Honest limits of the v0 read-then-check window: the request body is fully
read before the 20 MiB cap check — hostile clients can force a large read
(Starlette spools file parts over 1 MiB to disk, which bounds the memory
blast radius but not the read itself); a streaming/chunked cap is the
hardening follow-up. The Content-Length header check below is only a
best-effort early reject — the authoritative gate stays read-then-check
because a lying/absent header must not bypass it.
"""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.ingestion.import_formats import SOURCE_FORMATS, ImportFormatError
from app.mcp_hub.identity import AgentPrincipal, resolve_principal
from app.models.user import User
from app.schemas.Orivory import ImportSummary
from app.services.agent_token_service import TOKEN_PREFIX
from app.services.import_service import run_import
from app.utils.dependencies import get_current_user


async def _optional_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """The importing party, or None when the caller is an agent token.

    Routes on the Bearer VALUE: an ``oa_`` token is the agent path, which the
    endpoint below resolves to a principal plus its owner (so the import is
    scoped and ledgered under that agent). Anything else is a local request,
    which IS the local owner.
    """
    auth_header = request.headers.get("authorization", "") or ""
    scheme, _, value = auth_header.partition(" ")
    value = value.strip()
    if scheme.lower() == "bearer" and value.startswith(TOKEN_PREFIX):
        return None
    credentials = (
        HTTPAuthorizationCredentials(scheme="Bearer", credentials=value) if value else None
    )
    return await get_current_user(credentials, db)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/imports", tags=["imports"])

MAX_IMPORT_UPLOAD_BYTES = 20 * 1024 * 1024  # keeps the v0 path synchronous


async def _resolve_import_user(
    request: Request,
    db: AsyncSession,
) -> tuple[User | None, AgentPrincipal | None]:
    """Resolve the importing party: the local owner OR an agent token.

    Agent-token imports (the auto-capture path) attribute the memories to
    the agent's owner and record the agent in the ledger as an import —
    the same governance story as every other MCP write.
    """
    auth_header = request.headers.get("authorization", "") or ""
    scheme, _, value = auth_header.partition(" ")
    if scheme.lower() == "bearer" and value.strip().startswith("oa_"):
        principal = await resolve_principal(db, value.strip())
        if principal is None:
            return None, None
        from app.models.user import User as UserModel

        user = await db.get(UserModel, principal.user_id)
        return user, principal
    return None, None


@router.post("", response_model=ImportSummary, status_code=status.HTTP_201_CREATED)
async def create_import(
    current_user: Annotated[User | None, Depends(_optional_user)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
    request: Request = None,
    file: UploadFile = File(..., description="Export file (JSON): ChatGPT/Claude "
                                              "conversations.json, PAM memory-store.json, "
                                              "OpenClaw session dump, or a generic items array"),
    source_format: str | None = Form(default=None, description="One of: auto (default), "
                                                                "chatgpt, claude, gemini, copilot, openclaw, generic"),
) -> ImportSummary:
    """Import one export file as memories.

    Accepts TWO auth modes:
      - A local request (no token): the local owner — the self-host path.
      - An agent token (Authorization: Bearer oa_…) — the auto-capture path;
        memories attribute to the agent's owner and the import is ledgered
        as ``import`` by that agent.

    Auto-detects the format when ``source_format=auto`` (or is omitted).
    Idempotent per (user, format, ref): re-uploading a file skips
    already-imported items instead of duplicating them.
    """
    requested_by = "rest_api"
    agent_principal: AgentPrincipal | None = None

    if current_user is None:
        # No local identity — try the agent-token path before failing.
        user, agent_principal = await _resolve_import_user(request, db)
        if user is None:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required (invalid or revoked agent token).",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # Imports write memories: require the same scope as MCP writes.
        if not agent_principal.can_write():
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                detail="Agent token lacks the memory:write scope.",
            )
        current_user = user
        requested_by = f"agent:{agent_principal.name}"

    # A blank form field and an omitted one mean the same thing: auto-detect.
    # (curl's -F "source_format=" and stray client whitespace must not 422 —
    # strip first, THEN blankness-check, so " chatgpt " resolves normally.)
    if isinstance(source_format, str):
        source_format = source_format.strip() or None

    # Pre-validate an explicit format before reading/decoding anything —
    # an unknown value must fail here, not as a misleading "could not
    # detect" from the service (it IS detectable input that was mislabeled).
    if source_format is not None and source_format not in SOURCE_FORMATS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"unknown source_format: {source_format!r} (supported: {SOURCE_FORMATS})",
        )

    # Best-effort early reject on an over-cap Content-Length header: cheap
    # (no read), but not authoritative — the header can lie or be absent,
    # so the len(data) check below remains the gate.
    if request is not None:
        header = request.headers.get("content-length", "").strip()
        if header.isdigit() and int(header) > MAX_IMPORT_UPLOAD_BYTES:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                detail=(f"Content-Length {header} exceeds the "
                        f"{MAX_IMPORT_UPLOAD_BYTES // (1024 * 1024)} MiB synchronous import cap."),
            )

    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Uploaded file is empty.")
    if len(data) > MAX_IMPORT_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Import file exceeds the {MAX_IMPORT_UPLOAD_BYTES // (1024 * 1024)} MiB synchronous cap.",
        )

    # raw bytes, not a decoded str: run_import owns the utf-8 decode and
    # maps undecodable payloads to ImportFormatError → 422 (handoff 4).
    try:
        summary = await run_import(db, current_user.id, data, source_format, requested_by=requested_by)
    except ImportFormatError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    if agent_principal is not None:
        # Governance parity with MCP writes: agent-token imports are
        # recorded in the access ledger under the calling agent.
        from app.models.memory_access_log import MemoryAccessLog

        db.add(MemoryAccessLog(
            user_id=current_user.id,
            agent_client_id=agent_principal.agent_client_id,
            action="import",
            detail={"source_format": source_format, "requested_by": requested_by,
                    "created": summary.created, "file": file.filename or "upload"},
        ))
        await db.commit()
    log.info(
        "Import finished",
        extra={
            "user_id": str(current_user.id),
            "file": file.filename or "upload",
            "source_format": source_format,
            "requested_by": requested_by,
            # NOTE: never use `created` here — it collides with LogRecord's
            # reserved attribute and raises KeyError when the record emits.
            "created_count": summary.created,
            "skipped_duplicates": summary.skipped_duplicates,
            "suppressed_skipped": summary.suppressed_skipped,
        },
    )
    return summary


__all__ = ["MAX_IMPORT_UPLOAD_BYTES", "router"]
