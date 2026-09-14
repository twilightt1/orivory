"""Unit tests for the ErasureReceipt model: shape and defaults."""
from __future__ import annotations

import uuid

from app.models.erasure_receipt import ErasureReceipt


def test_erasure_receipt_defaults():
    receipt = ErasureReceipt(user_id=uuid.uuid4(), requested_memory_ids=[str(uuid.uuid4())])
    assert receipt.status == "completed"
    assert receipt.detail == {}
    assert len(receipt.requested_memory_ids) == 1


def test_erasure_receipt_empty_targets_default():
    receipt = ErasureReceipt(user_id=uuid.uuid4())
    assert receipt.requested_memory_ids == []
    assert receipt.status == "completed"


def test_erasure_receipt_status_column_fits_longest_status():
    """Regression: String(16) broke the INSERT exactly when the receipt mattered."""
    length = ErasureReceipt.__table__.columns["status"].type.length
    for status in ("completed", "completed_unverified", "completed_with_residual", "completed_with_errors"):
        assert len(status) <= length, f"{status!r} does not fit String({length})"
    assert length >= 32  # honest statuses ("completed_with_residual" = 23 chars) fit
