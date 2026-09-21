"""Custom SQLAlchemy column types."""
from __future__ import annotations

import json
import uuid as _uuid
from typing import Any

from sqlalchemy import JSON
from sqlalchemy import Uuid as _SaUuid
from sqlalchemy.types import TypeDecorator


class GUID(_SaUuid):
    """Cross-dialect UUID column.

    SQLAlchemy's native ``Uuid`` type rejects str-bound values on SQLite
    when the value arrives as a string (e.g. straight from Pydantic).
    This subclass coerces str → uuid.UUID before binding so the driver
    sees a real UUID object.
    """

    def bind_processor(self, dialect):
        impl_processor = super().bind_processor(dialect)

        def process(value):
            if isinstance(value, str):
                value = _uuid.UUID(value)
            return impl_processor(value)

        return process


class EncryptedJSONB(TypeDecorator):
    """A JSONB column whose dict value is encrypted at rest.

    On write, the Python dict is JSON-serialized, encrypted, and stored as a
    single JSON string. On read, the string is decrypted back into a dict.

    Backward compatibility: legacy rows that hold a plain JSON object (dict)
    instead of an encrypted string are returned as-is, so this can be rolled
    out without a data migration. New writes are always encrypted.
    """

    impl = JSON
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        from app.utils.crypto import encrypt_str

        # Store as an encrypted JSON string value inside the JSONB column.
        return encrypt_str(json.dumps(value, separators=(",", ":"), default=str))

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        # Legacy plaintext dict (pre-encryption rows) — pass through.
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            from app.utils.crypto import decrypt_str

            decrypted = decrypt_str(value)
            try:
                return json.loads(decrypted)
            except (json.JSONDecodeError, TypeError):
                return {}
        return value


__all__ = ["EncryptedJSONB"]
