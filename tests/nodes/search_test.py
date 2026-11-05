"""
tests/nodes/search_test.py

Search node against a fake es-mcp-server:
  - calls ``article_search`` by name with the state's query, categories and recency flag
  - returns the refs plus SEARCH timing
  - every failure shape (tool error, no structured content, non-text block, transport
    error) becomes ``SearchUnavailable`` with the fixed text only
"""

from __future__ import annotations

import pytest
from fastmcp.client.client import CallToolResult
from mcp_types import ImageContent, TextContent

from exception.search_unavailable import SEARCH_UNAVAILABLE_MESSAGE, SearchUnavailable
from nodes.search import Search
from tests.fake_mcp import FakeNewsMcp, StubMcpClient

pytestmark = pytest.mark.anyio


def _state(fetch_latest: bool = False) -> dict:
    return {
        "query": "ai in healthcare",
        "categories": ["technology", "health"],
        "fetch_latest": fetch_latest,
    }


class TestSearchCall:
    async def test_returns_refs_and_timing(self, fake_mcp: FakeNewsMcp) -> None:
        fake_mcp.refs = [
            {"article_id": "a1", "score": 2.0},
            {"article_id": "a2", "score": 1.0},
        ]
        out = await Search(fake_mcp.client())(_state())
        assert out["article_refs"] == fake_mcp.refs
        assert set(out["duration_seconds"]) == {"SEARCH"}

    async def test_sends_query_categories_and_recency(
        self, fake_mcp: FakeNewsMcp
    ) -> None:
        await Search(fake_mcp.client())(_state(fetch_latest=True))
        assert fake_mcp.calls == [
            (
                "article_search",
                {
                    "query": "ai in healthcare",
                    "categories": ["technology", "health"],
                    "fetch_latest": True,
                },
            )
        ]

    async def test_tool_error_is_search_unavailable(
        self, fake_mcp: FakeNewsMcp
    ) -> None:
        fake_mcp.fail = {"article_search"}
        with pytest.raises(SearchUnavailable) as exc:
            await Search(fake_mcp.client())(_state())
        assert str(exc.value) == SEARCH_UNAVAILABLE_MESSAGE

    async def test_unexpected_server_crash_is_search_unavailable(
        self, fake_mcp: FakeNewsMcp
    ) -> None:
        fake_mcp.crash = {"article_search"}
        with pytest.raises(SearchUnavailable) as exc:
            await Search(fake_mcp.client())(_state())
        assert str(exc.value) == SEARCH_UNAVAILABLE_MESSAGE
        assert "elasticsearch" not in str(exc.value)

    async def test_calls_tool_without_raising_on_error(self) -> None:
        stub = StubMcpClient(
            CallToolResult(
                content=[], structured_content={"article_refs": []}, meta=None
            )
        )
        await Search(stub)(_state())  # type: ignore[arg-type]
        assert stub.calls[0][0] == "article_search" and stub.calls[0][2] is False

    @pytest.mark.parametrize(
        "result",
        [
            CallToolResult(
                content=[TextContent(type="text", text="boom")],
                structured_content=None,
                meta=None,
                is_error=True,
            ),
            CallToolResult(content=[], structured_content=None, meta=None),
            CallToolResult(
                content=[ImageContent(type="image", data="AAAA", mimeType="image/png")],
                structured_content=None,
                meta=None,
            ),
        ],
        ids=["is_error", "no_structured_content", "non_text_block"],
    )
    async def test_bad_results_are_search_unavailable(
        self, result: CallToolResult
    ) -> None:
        with pytest.raises(SearchUnavailable) as exc:
            await Search(StubMcpClient(result))(_state())  # type: ignore[arg-type]
        assert str(exc.value) == SEARCH_UNAVAILABLE_MESSAGE

    async def test_transport_error_is_search_unavailable_and_chained(self) -> None:
        cause = ConnectionError("es-mcp-lb:9599 refused")
        with pytest.raises(SearchUnavailable) as exc:
            await Search(StubMcpClient(exc=cause))(_state())  # type: ignore[arg-type]
        assert str(exc.value) == SEARCH_UNAVAILABLE_MESSAGE
        assert exc.value.__cause__ is cause
