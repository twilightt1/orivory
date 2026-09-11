"""FastMCP server exposing the Orivory memory hub over stateless HTTP.

Verified pattern (mcp 1.29.1 installed; pyproject pins ``mcp>=1.9.0,<2``
because the 2.x line renamed ``FastMCP`` → ``MCPServer`` and dropped the
``stateless_http``/``json_response`` constructor flags — the SDK's own
migration note says to pin ``mcp<2`` for v1 code):

- ``FastMCP("orivory-memory", stateless_http=True, json_response=True)``:
  each HTTP request gets a fresh transport (no session persistence) and
  responses are plain JSON instead of SSE frames. Both flags are constructor
  parameters that flow into the instance settings.
- ``streamable_http_app()`` takes NO keyword arguments in 1.x; it reads
  ``stateless_http``/``json_response`` from the settings, creates the
  ``StreamableHTTPSessionManager`` lazily on first call, and returns a
  Starlette app with an inner route at ``settings.streamable_http_path``.
- ``streamable_http_path="/"`` (constructor kwarg) + ``FastAPI.mount("/mcp",
  get_mcp_app())`` serves the MCP endpoint at ``/mcp`` — the default inner
  path ``"/mcp"`` would yield ``/mcp/mcp`` under a mount. Starlette 1.6
  answers an exact ``POST /mcp`` (mount root) with 307 → ``/mcp/``, a hop
  that can drop the ``Authorization`` header — so ``main.py`` also registers
  the same app on an exact ``Route("/mcp", ...)``, and the :class:`McpAsgiApp`
  adapter normalizes the scope path so the SDK's inner ``Route("/")`` matches
  under both invocation styles (verified with an in-process
  initialize/tools/list/tools/call handshake).
- Starlette does NOT run a mounted app's lifespan: the host app's lifespan
  must enter ``mcp.session_manager.run()`` itself (the SDK docs' ASGI recap:
  "failure to do so will result in the first request failing"). The session
  manager only exists after ``streamable_http_app()`` has been called, which
  the mount at import time guarantees before startup.
- Transport security is explicit-or-default: without ``MCP_HUB_ALLOWED_HOSTS``
  the constructor passes ``transport_security=None`` and FastMCP (mcp 1.29.1)
  auto-enables localhost-only DNS-rebind protection because its default
  ``host`` is ``127.0.0.1`` (verified in the installed SDK source:
  ``FastMCP.__init__`` substitutes ``TransportSecuritySettings`` with
  ``allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"]`` — a
  non-localhost ``Host`` header gets 421). A non-empty CSV setting instead
  passes ``TransportSecuritySettings(enable_dns_rebinding_protection=True,
  allowed_hosts=[...])`` for reverse-proxy deployments.
- Tools receive the SDK ``Context`` via a parameter annotated ``Context``
  (excluded from the tool's input schema when it defaults to ``None``). The
  MCP protocol does not carry caller identity: each wrapper extracts the
  bearer / custom-header token from ``ctx.request_context.request.headers``
  and resolves it through ``resolve_principal`` on a short-lived session
  (which commits itself), then publishes the principal to the tool bodies via
  ``tools._principal_var`` — so the tool functions stay framework-free and
  tests can monkeypatch ``tools._current_principal`` instead. The token is
  never logged.
"""
from __future__ import annotations

import logging
from collections.abc import MutableMapping
from functools import lru_cache
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from app.config import settings
from app.database import AsyncSessionLocal
from app.mcp_hub import tools as hub_tools
from app.mcp_hub.identity import AgentPrincipal, extract_token, resolve_principal

log = logging.getLogger(__name__)


async def _principal_from_context(ctx: Context | None) -> AgentPrincipal | None:
    """Resolve the caller's principal from the MCP HTTP request headers.

    ``resolve_principal`` commits the short-lived session it is given; that is
    fine because the session is used for nothing else.
    """
    if ctx is None:
        return None
    request = getattr(ctx.request_context, "request", None)
    if request is None:
        return None
    token = extract_token(request.headers)
    if not token:
        return None
    async with AsyncSessionLocal() as db:
        return await resolve_principal(db, token)


async def _call_with_identity(coro: Any, ctx: Context | None) -> dict[str, Any]:
    """Publish the resolved principal to the tool body, then await it."""
    principal = await _principal_from_context(ctx)
    var_token = hub_tools._principal_var.set(principal)
    try:
        return await coro
    finally:
        hub_tools._principal_var.reset(var_token)


def resolve_transport_security(allowed_hosts_csv: str) -> TransportSecuritySettings | None:
    """Build the FastMCP ``transport_security`` from a CSV env value.

    Empty (or blank-only entries) → ``None``: FastMCP is constructed without
    explicit transport security and its own default applies — with the
    default ``host="127.0.0.1"``, the SDK auto-enables localhost-only
    DNS-rebind protection (non-localhost ``Host`` → 421), which is the
    shipped behaviour for this app.

    A non-empty CSV → explicit ``TransportSecuritySettings`` with rebind
    protection on and exactly the given hosts allowed, e.g. behind a reverse
    proxy that forwards a public ``Host`` header.
    """
    hosts = [h.strip() for h in allowed_hosts_csv.split(",") if h.strip()]
    if not hosts:
        return None
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
    )


def _build_server() -> FastMCP:
    mcp = FastMCP(
        "orivory-memory",
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
        transport_security=resolve_transport_security(settings.MCP_HUB_ALLOWED_HOSTS),
    )

    @mcp.tool()
    async def search_memory(query: str, limit: int = 8, include_history: bool = False, ctx: Context = None) -> dict[str, Any]:
        """Search the user's memory hub — returns an INDEX (id/title/snippet),
        NOT full content. Progressive disclosure: review the index, then call
        get_memory on the few ids that matter or timeline for context around
        one. Do NOT fetch details for every hit (requires memory:read)."""
        return await _call_with_identity(hub_tools.search_memory(query=query, limit=limit, include_history=include_history), ctx)

    @mcp.tool()
    async def timeline(memory_id: str, window: int = 4, ctx: Context = None) -> dict[str, Any]:
        """Context AROUND one memory: the anchor plus up to `window`
        neighbours captured before and after it (memory:read). Cheaper than
        get_memory on many ids when you need surrounding history."""
        return await _call_with_identity(
            hub_tools.timeline(memory_id=memory_id, window=window), ctx
        )

    @mcp.tool()
    async def get_memory(memory_id: str, ctx: Context = None) -> dict[str, Any]:
        """Fetch one memory by id, owned by the caller (memory:read)."""
        return await _call_with_identity(hub_tools.get_memory(memory_id=memory_id), ctx)

    @mcp.tool()
    async def list_recent(limit: int = 20, ctx: Context = None) -> dict[str, Any]:
        """List the caller's most recent memories (memory:read)."""
        return await _call_with_identity(hub_tools.list_recent(limit=limit), ctx)

    @mcp.tool()
    async def add_memory(
        title: str, content: str, tags: list[str] | None = None, ctx: Context = None
    ) -> dict[str, Any]:
        """Store a new memory owned by the caller (memory:write)."""
        return await _call_with_identity(hub_tools.add_memory(title=title, content=content, tags=tags), ctx)

    @mcp.tool()
    async def correct_memory(
        content: str, title: str = "", subject: str = "", attribute: str = "",
        scope: str = "default", memory_id: str | None = None,
        valid_from: str | None = None, evidence_ids: list[str] | None = None,
        ctx: Context = None,
    ) -> dict[str, Any]:
        """Correct a stored fact with evidence (memory:write). Creates a linked new version; never overwrites."""
        return await _call_with_identity(hub_tools.correct_memory(
            memory_id=memory_id, subject=subject, attribute=attribute, scope=scope,
            title=title, content=content, valid_from=valid_from,
            evidence_ids=evidence_ids), ctx)

    @mcp.tool()
    async def delete_memory(memory_id: str, ctx: Context = None) -> dict[str, Any]:
        """Delete one memory owned by the caller (memory:write)."""
        return await _call_with_identity(hub_tools.delete_memory(memory_id=memory_id), ctx)

    @mcp.tool()
    async def forget_memory(memory_ids: list[str], ctx: Context = None) -> dict[str, Any]:
        """Erase memories and derived artifacts with a verification receipt (memory:write)."""
        return await _call_with_identity(hub_tools.forget_memory(memory_ids=memory_ids), ctx)

    return mcp


mcp = _build_server()


def build_mcp_server() -> FastMCP:
    """Return the module's FastMCP instance with the seven tools registered.

    A single instance is shared with :func:`get_mcp_app` so the mounted app
    and the host lifespan drive the same session manager.
    """
    return mcp


def normalize_mcp_scope(scope: MutableMapping) -> None:
    """Rewrite an HTTP scope so the SDK's inner ``Route("/")`` matches.

    The host app reaches the MCP endpoint two ways: the ``/mcp`` mount
    (scope path ``/mcp/...`` with ``root_path=/mcp``) and the exact
    ``Route("/mcp", ...)`` (scope path ``/mcp``). In both cases the path
    relative to ``root_path`` must collapse to ``/`` for the inner route;
    Starlette 1.6's exact-root 307 redirect to ``/mcp/`` would otherwise
    strip the ``Authorization`` header. Mutates the scope in place.
    """
    root = scope.get("root_path", "")
    path = scope.get("path", "")
    sub_path = path[len(root):] if path.startswith(root) else path
    if sub_path != "/":
        scope["path"] = root + "/"


class McpAsgiApp:
    """ASGI adapter presenting the SDK streamable HTTP app at /mcp and /mcp/.

    A callable *instance* (not a function) so Starlette treats it as a raw
    ASGI endpoint for every HTTP method; the path rewrite itself lives in
    :func:`normalize_mcp_scope`.
    """

    def __init__(self, sdk_app: Any) -> None:
        self._sdk_app = sdk_app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") == "http":
            normalize_mcp_scope(scope)
        await self._sdk_app(scope, receive, send)


@lru_cache(maxsize=1)
def get_mcp_app():
    """ASGI app for the MCP endpoint (module-cached; stateless streamable HTTP)."""
    return McpAsgiApp(mcp.streamable_http_app())


__all__ = ["build_mcp_server", "get_mcp_app"]
