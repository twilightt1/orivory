"""Regression tests for FastAPI dependency resolution on routers.

These tests catch the class of bug where a router uses ``Annotated`` in
endpoint signatures without importing it. Combined with
``from __future__ import annotations``, FastAPI cannot resolve the string
annotations and silently registers auth/db dependencies as *query params*,
meaning the endpoints run unauthenticated (or error) at runtime.

Seen in the field: a router module that omitted ``from typing import Annotated``.
"""
from __future__ import annotations

import importlib
import typing

import pytest
from fastapi.routing import APIRoute

from app.main import app

ROUTERS_TO_CHECK = [
    "app.api.v1.memories",
    "app.api.v1.agents",
    "app.api.v1.erasure",
    "app.api.v1.imports",
    "app.api.v1.users",
]


@pytest.mark.parametrize("module_name", ROUTERS_TO_CHECK)
def test_router_routes_have_resolvable_type_hints(module_name: str):
    """Every route must expose resolvable type hints (no unresolved ForwardRef)."""
    module = importlib.import_module(module_name)
    router = getattr(module, "router", None)
    if router is None:
        pytest.skip(f"{module_name} has no router")
    for route in router.routes:
        try:
            typing.get_type_hints(route.endpoint)
        except NameError as exc:
            pytest.fail(f"{module_name} route {route.path}: unresolvable annotations — {exc}")


@pytest.mark.parametrize("module_name", ROUTERS_TO_CHECK)
def test_router_dependencies_are_not_query_params(module_name: str):
    """Auth/session dependencies must be registered as dependencies, not query params."""
    module = importlib.import_module(module_name)
    router = getattr(module, "router", None)
    if router is None:
        pytest.skip(f"{module_name} has no router")
    for route in router.routes:
        dependant = route.dependant
        # Session-scoped dependencies (db/current_user) must never appear as
        # query params — that means FastAPI failed to resolve the annotation.
        query_names = {p.name for p in dependant.query_params}
        for forbidden in ("current_user", "db"):
            assert forbidden not in query_names, (
                f"{module_name} route {route.path}: '{forbidden}' was registered as a "
                f"query param — dependency annotation is unresolved "
                f"(missing import of Annotated?)"
            )


def test_no_unresolved_forward_refs_in_registered_routes():
    """Walk every registered APIRoute and resolve its endpoint type hints."""
    unresolved = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        try:
            typing.get_type_hints(route.endpoint)
        except NameError as exc:
            unresolved.append((route.path, str(exc)))
    assert not unresolved, f"Routes with unresolved annotations: {unresolved}"
