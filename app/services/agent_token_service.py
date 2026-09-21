"""Token generation/hashing for Open Memory Hub agent clients.

Plaintext tokens look like ``oa_<32 hex>`` and are shown exactly once, in
the registration response. Only ``sha256`` digests are persisted, so a
database leak never leaks usable credentials. Log lines may carry the
8-char digest prefix (``token_hash_prefix``) for support correlation.
"""
from __future__ import annotations

import hashlib
import secrets

ALLOWED_SCOPES = ("memory:read", "memory:write")
DEFAULT_SCOPES: tuple[str, ...] = ("memory:read",)

# What every plaintext agent token starts with. The identity layer keys its
# "this Bearer IS an agent token" branch on it (app/utils/dependencies.py), so
# the generator and the resolver cannot drift apart.
TOKEN_PREFIX = "oa_"


def generate_token() -> str:
    return TOKEN_PREFIX + secrets.token_hex(16)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_hash_prefix(token: str) -> str:
    return hash_token(token)[:8]


def validate_scopes(scopes: list[str]) -> list[str]:
    if not scopes:
        return list(DEFAULT_SCOPES)
    for scope in scopes:
        if scope not in ALLOWED_SCOPES:
            raise ValueError(f"unknown scope: {scope!r} (allowed: {ALLOWED_SCOPES})")
    return list(dict.fromkeys(scopes))


__all__ = ["ALLOWED_SCOPES", "DEFAULT_SCOPES", "TOKEN_PREFIX", "generate_token", "hash_token", "token_hash_prefix", "validate_scopes"]
