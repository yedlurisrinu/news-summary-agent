"""
tests/graph/build_graph_test.py

The compiled workflow (classify → search → summarize, two guard rails) against a fake
LLM, a fake es-mcp-server and an in-memory checkpointer:
  - routers pick the right branch
  - news query → summary from the LLM with the fetched articles
  - off-topic → guard rail #1, no MCP calls
  - no search results → guard rail #2, mget never called
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from data.seed_data import SeedData
from graph.build_graph import build, route_after_classify, route_after_search
from tests.fake_mcp import FakeNewsMcp

pytestmark = pytest.mark.anyio

NEWS = json.dumps(
    {"is_news_query": True, "categories": ["technology"], "fetch_latest": False}
)
NOT_NEWS = json.dumps({"is_news_query": False, "categories": [], "fetch_latest": False})


class TestRouters:
    async def test_route_after_classify_news(self) -> None:
        assert (
            await route_after_classify({"query": "q", "is_news_query": True})
            == "search"
        )  # type: ignore[typeddict-item]

    async def test_route_after_classify_off_topic(self) -> None:
        assert (
            await route_after_classify({"query": "q", "is_news_query": False})
            == "reply_off_topic"
        )  # type: ignore[typeddict-item]

    async def test_route_after_search_found(self) -> None:
        state = {"query": "q", "article_refs": [{"article_id": "a1", "score": 1.0}]}
        assert await route_after_search(state) == "found"  # type: ignore[arg-type]

    async def test_route_after_search_empty(self) -> None:
        assert await route_after_search({"query": "q", "article_refs": []}) == "empty"  # type: ignore[typeddict-item]


async def _run(llm, fake_mcp: FakeNewsMcp, query: str) -> dict:
    graph = await build(InMemorySaver(), llm, fake_mcp.client())
    return await graph.ainvoke(
        {"query": query}, {"configurable": {"thread_id": uuid4().hex}}
    )


class TestEndToEndRouting:
    async def test_normal_path_produces_summary(
        self, make_llm, fake_mcp: FakeNewsMcp
    ) -> None:
        out = await _run(make_llm([NEWS, "Grounded summary."]), fake_mcp, "tech news")
        assert out["summary"] == "Grounded summary."
        assert [h["article_id"] for h in out["article_hits"]] == ["a1"]
        assert [name for name, _ in fake_mcp.calls] == [
            "category_search",
            "article_search",
            "mget",
        ]

    async def test_off_topic_hits_guard_rail_1(
        self, make_llm, fake_mcp: FakeNewsMcp
    ) -> None:
        out = await _run(make_llm([NOT_NEWS]), fake_mcp, "write a poem")
        assert out["summary"] == SeedData._OFF_TOPIC_REPLY
        assert out["article_hits"] == [] and fake_mcp.calls == []

    async def test_no_results_hits_guard_rail_2(
        self, make_llm, fake_mcp: FakeNewsMcp
    ) -> None:
        fake_mcp.refs = []
        out = await _run(make_llm([NEWS]), fake_mcp, "tech news")
        assert out["summary"] == SeedData._NO_NEWS_FOUND_REPLY
        assert "mget" not in [name for name, _ in fake_mcp.calls]
