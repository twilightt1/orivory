"""
Erasure receipts API — verifiable memory erasure (Open Memory Hub MVP 5).

Endpoints:
    POST /api/v1/erasure-receipts          erase memories owned by the caller
    GET  /api/v1/erasure-receipts          list receipts, newest first, user-scoped
    GET  /api/v1/erasure-receipts/{id}     fetch one receipt

Foreign/unknown memory ids are recorded in the receipt as
``not_found_or_foreign`` — never deleted, no existence leak. Receipts quote
personal memory ids, so they cascade away with the user.
"""
from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.erasure_receipt import ErasureReceipt
from app.models.user import User
from app.schemas.Orivory import (
    ErasureReceiptCreate,
    ErasureReceiptItem,
    ErasureReceiptListResponse,
)
from app.services.erasure_service import erase_memories, list_receipts
from app.utils.dependencies import get_current_user

router = APIRouter(prefix="/erasure-receipts", tags=["erasure"])


@router.post("", response_model=ErasureReceiptItem, status_code=status.HTTP_201_CREATED)
async def create_erasure_receipt(
    body: ErasureReceiptCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ErasureReceiptItem:
    """Erase the given memories (caller-owned only) and return the receipt."""
    receipt = await erase_memories(db, current_user.id, body.memory_ids, requested_by="rest_api")
    return ErasureReceiptItem.model_validate(receipt)


@router.get("", response_model=ErasureReceiptListResponse)
async def list_erasure_receipts(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> ErasureReceiptListResponse:
    """Erasure receipts for the current user, newest first."""
    rows, total = await list_receipts(db, current_user.id, limit=limit, offset=offset)
    return ErasureReceiptListResponse(
        items=[ErasureReceiptItem.model_validate(row) for row in rows],
        total=total,
    )


@router.get("/{receipt_id}", response_model=ErasureReceiptItem)
async def get_erasure_receipt(
    receipt_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ErasureReceiptItem:
    """Fetch one receipt. Foreign or unknown ids read as 404 (no existence leak)."""
    receipt = await db.get(ErasureReceipt, receipt_id)
    if not receipt or receipt.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Erasure receipt not found.")
    return ErasureReceiptItem.model_validate(receipt)
