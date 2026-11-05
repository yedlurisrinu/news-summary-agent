"""
tests/adapters/mcp_session_test.py

LazyMcpSession (one es-mcp-server connection, opened on first use and reused):
  - nothing connects until the first tool call (startup independence)
  - one connection for many calls, including concurrent first calls
  - a failed connect is retried on the next call
  - a dropped connection is reopened; aclose() disconnects only if connected
  - works end to end with the graph nodes against the in-memory fake es-mcp-server
"""

from __future__ import annotations

import asyncio
from typing import Any, Self

import pytest
from fastmcp.client.client import CallToolResult

from adapters.mcp_session import LazyMcpSession
from nodes.search import Search
from tests.fake_mcp import FakeNewsMcp

pytestmark = pytest.mark.anyio

RESULT = CallToolResult(content=[], structured_content={"article_refs": []}, meta=None)


class FakeClient:
    """Counts connects/disconnects; can fail to connect or report a dropped connection."""

    def __init__(self, connect_failures: int = 0) -> None:
        self.enters = 0
        self.exits = 0
        self.calls = 0
        self.connect_failures = connect_failures
        self.connected = False

    async def __aenter__(self) -> Self:
        await asyncio.sleep(0)  # let concurrent callers race for the lock
        if self.connect_failures:
            self.connect_failures -= 1
            raise ConnectionError("es-mcp-lb unreachable")
        self.enters += 1
        self.connected = True
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.exits += 1
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    async def call_tool(self, name: str, arguments: dict[str, Any], raise_on_error: bool = True) -> CallToolResult:
        assert self.connected, "call_tool on a closed connection"
        self.calls += 1
        return RESULT


async def test_nothing_connects_until_first_call() -> None:
    client = FakeClient()
    session = LazyMcpSession(client)  # type: ignore[arg-type]
    async with session:  # what the nodes do: must not connect either
        pass
    assert client.enters == 0
    await session.call_tool("article_search", {})
    assert client.enters == 1


async def test_connection_is_reused_across_calls() -> None:
    client = FakeClient()
    session = LazyMcpSession(client)  # type: ignore[arg-type]
    for _ in range(5):
        async with session:
            await session.call_tool("article_search", {})
    assert (client.enters, client.calls, client.exits) == (1, 5, 0)


async def test_concurrent_first_calls_connect_once() -> None:
    client = FakeClient()
    session = LazyMcpSession(client)  # type: ignore[arg-type]
    await asyncio.gather(*(session.call_tool("mget", {}) for _ in range(10)))
    assert client.enters == 1 and client.calls == 10


async def test_failed_connect_is_retried_on_next_call() -> None:
    client = FakeClient(connect_failures=1)
    session = LazyMcpSession(client)  # type: ignore[arg-type]
    with pytest.raises(ConnectionError):
        await session.call_tool("article_search", {})
    assert await session.call_tool("article_search", {}) is RESULT
    assert client.enters == 1


async def test_dropped_connection_is_reopened() -> None:
    client = FakeClient()
    session = LazyMcpSession(client)  # type: ignore[arg-type]
    await session.call_tool("article_search", {})
    client.connected = False  # the background session died
    await session.call_tool("article_search", {})
    assert client.enters == 2 and client.exits == 1


async def test_aclose_disconnects_only_if_connected() -> None:
    never_used = FakeClient()
    await LazyMcpSession(never_used).aclose()  # type: ignore[arg-type]
    assert never_used.exits == 0

    used = FakeClient()
    session = LazyMcpSession(used)  # type: ignore[arg-type]
    await session.call_tool("article_search", {})
    await session.aclose()
    assert used.exits == 1


async def test_close_error_is_logged_not_raised() -> None:
    client = FakeClient()
    session = LazyMcpSession(client)  # type: ignore[arg-type]
    await session.call_tool("article_search", {})

    async def broken_exit(*exc_info: Any) -> None:
        raise RuntimeError("socket already gone")

    client.__aexit__ = broken_exit  # type: ignore[method-assign]
    await session.aclose()  # must not raise


async def test_graph_node_through_lazy_session(fake_mcp: FakeNewsMcp) -> None:
    session = LazyMcpSession(fake_mcp.client())
    search = Search(session)
    state = {"query": "q", "categories": ["technology"], "fetch_latest": False}
    first = await search(state)  # type: ignore[arg-type]
    second = await search(state)  # type: ignore[arg-type]
    assert first["article_refs"] == second["article_refs"] == fake_mcp.refs
    await session.aclose()
