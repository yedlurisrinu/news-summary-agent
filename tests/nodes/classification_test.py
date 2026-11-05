"""
tests/nodes/classification_test.py

ClassifyIntent node with a fake LLM and a fake es-mcp-server:
  - non-news queries short-circuit (no MCP call)
  - news queries call ``category_search`` with the extracted categories and record timings
  - a failing category check never fails the node: classification proceeds
"""

from __future__ import annotations

import json

import pytest
from fastmcp.client.client import CallToolResult
from mcp_types import ImageContent

from nodes.classification import ClassifyIntent
from tests.fake_mcp import FakeNewsMcp, StubMcpClient

pytestmark = pytest.mark.anyio

NEWS = json.dumps(
    {"is_news_query": True, "categories": ["technology"], "fetch_latest": True}
)
NOT_NEWS = json.dumps({"is_news_query": False, "categories": [], "fetch_latest": False})


class TestClassifyIntentCall:
    async def test_non_news_query_short_circuits(
        self, make_llm, fake_mcp: FakeNewsMcp
    ) -> None:
        out = await ClassifyIntent(make_llm([NOT_NEWS]), fake_mcp.client())(
            {"query": "write a poem"}
        )
        assert out["is_news_query"] is False
        assert fake_mcp.calls == []

    async def test_news_query_resolves_categories_and_times(
        self, make_llm, fake_mcp: FakeNewsMcp
    ) -> None:
        out = await ClassifyIntent(make_llm([NEWS]), fake_mcp.client())(
            {"query": "latest tech news"}
        )
        assert out["is_news_query"] is True and out["fetch_latest"] is True
        assert out["categories"] == ["technology"]
        assert fake_mcp.calls == [("category_search", {"categories": ["technology"]})]
        assert set(out["duration_seconds"]) == {"LLM_LATENCY", "ES_CALL"}

    async def test_category_tool_error_is_swallowed(
        self, make_llm, fake_mcp: FakeNewsMcp
    ) -> None:
        fake_mcp.fail = {"category_search"}
        out = await ClassifyIntent(make_llm([NEWS]), fake_mcp.client())(
            {"query": "tech news"}
        )
        assert out["is_news_query"] is True and out["categories"] == ["technology"]

    async def test_resolve_categories_returns_hit_count(
        self, make_llm, fake_mcp: FakeNewsMcp
    ) -> None:
        fake_mcp.category_hits = 7
        node = ClassifyIntent(make_llm([NEWS]), fake_mcp.client())
        assert await node._resolve_categories(["technology"]) == 7

    async def test_zero_hits_is_reported_as_zero(
        self, make_llm, fake_mcp: FakeNewsMcp
    ) -> None:
        fake_mcp.category_hits = 0
        node = ClassifyIntent(make_llm([NEWS]), fake_mcp.client())
        assert await node._resolve_categories(["technology"]) == 0

    @pytest.mark.parametrize(
        "stub",
        [
            StubMcpClient(exc=ConnectionError("es-mcp-lb down")),
            StubMcpClient(
                CallToolResult(content=[], structured_content=None, meta=None)
            ),
            StubMcpClient(
                CallToolResult(
                    content=[
                        ImageContent(type="image", data="AAAA", mimeType="image/png")
                    ],
                    structured_content=None,
                    meta=None,
                    is_error=True,
                )
            ),
        ],
        ids=["transport_error", "no_structured_content", "non_text_block"],
    )
    async def test_failed_category_check_counts_as_zero(
        self, make_llm, stub: StubMcpClient
    ) -> None:
        node = ClassifyIntent(make_llm([NEWS]), stub)  # type: ignore[arg-type]
        assert await node._resolve_categories(["technology"]) == 0
