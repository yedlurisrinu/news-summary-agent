"""Test doubles for the es-mcp-server tools the graph nodes call (no network, no ES).

``FakeNewsMcp`` is a real in-memory MCP server (fastmcp) with the same tool names and
argument shapes as es-mcp-server: ``category_search``, ``article_search`` and ``mget``.
Nodes talk to it through ``fastmcp.Client(server)``, exactly as they talk to the real
server over HTTP. Each tool can be switched to fail with a ``ToolError`` (an expected,
fixed-text failure) or to crash with an unexpected exception.

``StubMcpClient`` returns one hand-built ``CallToolResult`` (or raises), for shapes a
real server rarely produces: no structured content, or a non-text content block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

from fastmcp import Client, FastMCP
from fastmcp.client.client import CallToolResult
from fastmcp.exceptions import ToolError
from pydantic import BaseModel

NO_ARTICLES_FOUND = "No articles found for the given client ids"


class NewsQuery(BaseModel):
    query: str
    categories: list[str]
    fetch_latest: bool


def make_doc(article_id: str, content: str = "body text") -> dict[str, Any]:
    """One article as es-mcp-server's ``mget`` returns it."""
    return {
        "article_id": article_id,
        "title": f"Title {article_id}",
        "source": "TechNews",
        "link": f"https://technews.example.com/{article_id}",
        "published_at": "2026-11-01T10:00:00Z",
        "content": content,
    }


@dataclass
class FakeNewsMcp:
    category_hits: int = 3
    refs: list[dict[str, Any]] = field(
        default_factory=lambda: [{"article_id": "a1", "score": 1.5}]
    )
    docs: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {"a1": make_doc("a1")}
    )
    fail: set[str] = field(default_factory=set)  # tools raising ToolError
    crash: set[str] = field(default_factory=set)  # tools raising RuntimeError
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def _check(self, tool: str, args: dict[str, Any]) -> None:
        self.calls.append((tool, args))
        if tool in self.crash:
            raise RuntimeError("internal detail: es host elasticsearch:9200")
        if tool in self.fail:
            raise ToolError(f"{tool} is unavailable")

    def server(self) -> FastMCP:
        mcp = FastMCP("fake-es-mcp-server")

        @mcp.tool
        def category_search(categories: list[str]) -> int:
            self._check("category_search", {"categories": categories})
            return self.category_hits

        @mcp.tool
        def article_search(user_query: NewsQuery) -> dict[str, Any]:
            self._check("article_search", user_query.model_dump())
            return {"article_refs": self.refs}

        @mcp.tool
        def mget(ids: list[str]) -> dict[str, Any]:
            self._check("mget", {"ids": ids})
            hits = [self.docs[i] for i in ids if i in self.docs]
            if not hits:
                raise ToolError(NO_ARTICLES_FOUND)
            return {"article_hits": hits, "content_length": 1000, "truncated": True}

        return mcp

    def client(self) -> Client[Any]:
        return Client(self.server())


class StubMcpClient:
    """Minimal stand-in for ``fastmcp.Client``: one canned result or exception."""

    def __init__(
        self, result: CallToolResult | None = None, exc: Exception | None = None
    ) -> None:
        self._result = result
        self._exc = exc
        self.calls: list[tuple[str, dict[str, Any], bool]] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def call_tool(
        self, name: str, arguments: dict[str, Any], raise_on_error: bool = True
    ) -> CallToolResult:
        self.calls.append((name, arguments, raise_on_error))
        if self._exc is not None:
            raise self._exc
        assert self._result is not None
        return self._result
