"""Minimal wiring tests for the MCP server module (import-only, no network)."""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette._utils import get_route_path

from app.mcp_hub.server import build_mcp_server, get_mcp_app, normalize_mcp_scope, resolve_transport_security


def test_build_mcp_server_returns_named_fastmcp():
    server = build_mcp_server()
    assert isinstance(server, FastMCP)
    assert server.name == "orivory-memory"
    assert server.settings.stateless_http is True
    assert server.settings.json_response is True


def test_get_mcp_app_returns_asgi_app():
    app = get_mcp_app()
    assert callable(app)


def test_normalize_scope_exact_root_matches_inner_route():
    """Exact-route invocation (path=/mcp, no root_path): the SDK's inner ``Route("/")``
    must see ``/`` after the rewrite, without the 307 that would strip Authorization."""
    scope = {"type": "http", "path": "/mcp", "root_path": ""}
    normalize_mcp_scope(scope)
    assert get_route_path(scope) == "/"


def test_normalize_scope_mount_root_matches_inner_route():
    """Mount invocation (root_path=/mcp): the SDK's inner ``Route("/")`` must see ``/``."""
    scope = {"type": "http", "path": "/mcp", "root_path": "/mcp"}
    normalize_mcp_scope(scope)
    assert scope["path"] == "/mcp/"
    assert get_route_path(scope) == "/"


def test_normalize_scope_proxy_root_matches_inner_route():
    """Proxy-style root (root_path=/api): the rewrite keeps the prefix so the inner
    route still resolves to ``/`` — root_path is stripped by ``get_route_path``."""
    scope = {"type": "http", "path": "/api/mcp", "root_path": "/api"}
    normalize_mcp_scope(scope)
    assert get_route_path(scope) == "/"


def test_build_mcp_server_keeps_localhost_transport_default():
    """Empty MCP_HUB_ALLOWED_HOSTS (the shipped default) → no explicit
    transport_security is passed, so FastMCP's default applies: host stays
    127.0.0.1 and the SDK auto-enables localhost-only DNS-rebind protection
    (a non-localhost Host header is answered 421). Verified observable state
    on the FastMCP object (mcp 1.29.1): settings.transport_security is NOT
    None — it is the localhost auto-enabled TransportSecuritySettings."""
    server = build_mcp_server()
    assert server.settings.host == "127.0.0.1"
    ts = server.settings.transport_security
    assert isinstance(ts, TransportSecuritySettings)
    assert ts.enable_dns_rebinding_protection is True
    assert ts.allowed_hosts == ["127.0.0.1:*", "localhost:*", "[::1]:*"]


def test_resolve_transport_security_empty_returns_none():
    """Empty CSV → None → FastMCP default (localhost-only auto-protection)."""
    assert resolve_transport_security("") is None
    assert resolve_transport_security(" , ,") is None


def test_resolve_transport_security_csv_enables_protection_with_hosts():
    """Non-empty CSV → explicit rebind protection with exactly the parsed hosts."""
    ts = resolve_transport_security("a.example, b.example")
    assert isinstance(ts, TransportSecuritySettings)
    assert ts.enable_dns_rebinding_protection is True
    assert ts.allowed_hosts == ["a.example", "b.example"]


async def test_build_mcp_server_registers_forget_tool():
    server = build_mcp_server()
    tools = await server.list_tools()
    assert any(t.name == "forget_memory" for t in tools)
    assert any(t.name == "correct_memory" for t in tools)
