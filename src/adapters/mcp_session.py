"""MCP client adapter — one connection to es-mcp-server, opened on first use and reused.

Why lazy: startup must not depend on es-mcp-server or Keycloak. Nothing connects until the
first tool call; if that connect fails, the call fails and the next call tries again. Each
node handles a connect failure like any other tool failure: classify ignores it (count 0),
search raises ``SearchUnavailable`` (HTTP 503), summary replies with fixed text (HTTP 200).

Why reused: a fresh fastmcp connection per call costs two extra requests — the protocol
probe (``server/discover``, avoided by pinning the version) and ``tools/list`` (the mcp
client validates results against each tool's output schema, cached per connection). Reusing
one connection leaves one request per tool call: ``tools/call``.

The protocol (MCP 2026-07-28) is sessionless: every request is independent HTTP with no
server-side session, so an es-mcp-server restart does not invalidate the kept connection.
"""

from __future__ import annotations

import asyncio
import logging
from types import TracebackType
from typing import Any, Protocol, Self

from fastmcp import Client
from fastmcp.client.client import CallToolResult

logger = logging.getLogger(__name__)


class McpToolClient(Protocol):
    """What the graph nodes need from an MCP client (``fastmcp.Client`` satisfies it)."""

    async def __aenter__(self) -> Any: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Any: ...

    async def call_tool(
        self, name: str, arguments: dict[str, Any], raise_on_error: bool = True
    ) -> CallToolResult: ...


class LazyMcpSession:
    """Wraps a ``fastmcp.Client``: connects on the first ``call_tool``, then keeps the connection.

    ``async with session:`` is a no-op (the nodes keep their usual shape); ``aclose()``
    disconnects at app shutdown. A dropped connection is reopened on the next call.

    TODO(future fix): output schemas are cached for the lifetime of a connection (the mcp
    client fetches them once with ``tools/list`` to validate results). If a tool's output
    schema changes when es-mcp-server is redeployed, results could fail validation until the
    app restarts. The tool contracts are fixed today, so this is noted rather than handled.
    A fix would drop and reopen the connection when a result fails schema validation.
    """

    def __init__(self, client: Client[Any]) -> None:
        self._client = client
        self._connected = False
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None  # keep the connection for the next call

    async def _ensure_connected(self) -> None:
        if self._connected and self._client.is_connected():
            return
        async with self._lock:
            if self._connected and self._client.is_connected():
                return  # another call connected while we waited
            if self._connected:
                logger.warning("MCP connection dropped; reconnecting")
                await self._disconnect()
            await self._client.__aenter__()
            self._connected = True
            logger.info("MCP connection opened")

    async def call_tool(
        self, name: str, arguments: dict[str, Any], raise_on_error: bool = True
    ) -> CallToolResult:
        await self._ensure_connected()
        return await self._client.call_tool(
            name, arguments, raise_on_error=raise_on_error
        )

    async def _disconnect(self) -> None:
        self._connected = False
        try:
            await self._client.__aexit__(None, None, None)
        except Exception:
            logger.warning("error while closing the MCP connection", exc_info=True)

    async def aclose(self) -> None:
        """Disconnect at app shutdown (no-op if never connected)."""
        if self._connected:
            await self._disconnect()
