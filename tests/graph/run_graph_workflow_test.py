"""
tests/graph/run_graph_workflow_test.py

run_graph(): final state → QuerySuccessResponse, and failures → QueryFailedResponse with
fixed text only (never the exception's text):
  - SearchUnavailable → SEARCH_UNAVAILABLE_MESSAGE, error_code 503
  - anything else     → QUERY_FAILED_MESSAGE, error_code 500
  - each request gets a fresh thread_id, so repeated queries don't accumulate hits
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from data.news_input import NewsInput
from data.query_failed_response import QueryFailedResponse
from data.query_success_response import QuerySuccessResponse
from exception.search_unavailable import SEARCH_UNAVAILABLE_MESSAGE, SearchUnavailable
from graph.build_graph import build
from graph.run_graph_workflow import QUERY_FAILED_MESSAGE, cfg, run_graph
from tests.fake_mcp import FakeNewsMcp

pytestmark = pytest.mark.anyio

FINAL_STATE = {
    "query": "tech news",
    "categories": ["technology"],
    "summary": "S",
    "article_hits": [
        {"title": "T", "link": "https://x", "source": "TechNews", "published_at": "2026-11-01"}
    ],
    "duration_seconds": {"LLM_LATENCY": 0.5, "SEARCH": 0.25},
}


def _graph(result: object = None, exc: Exception | None = None) -> MagicMock:
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value=result, side_effect=exc)
    return graph


class TestRunGraph:
    def test_cfg_sets_thread_id_and_recursion_limit(self) -> None:
        assert cfg("t1", 10) == {"configurable": {"thread_id": "t1"}, "recursion_limit": 10}

    async def test_success_maps_state_to_response(self) -> None:
        out = await run_graph(NewsInput(query="tech news"), _graph(FINAL_STATE))
        assert isinstance(out, QuerySuccessResponse)
        assert out.summary == "S" and out.categories == ["technology"]
        assert out.articles[0].title == "T" and out.duration_seconds == 0.75

    async def test_search_unavailable_maps_to_fixed_text(self) -> None:
        out = await run_graph(NewsInput(query="q"), _graph(exc=SearchUnavailable()))
        assert isinstance(out, QueryFailedResponse)
        assert out.error == SEARCH_UNAVAILABLE_MESSAGE and out.error_code == 503

    async def test_unexpected_exception_hides_internals(self) -> None:
        exc = ConnectionError("postgres:5432 password authentication failed")
        out = await run_graph(NewsInput(query="q"), _graph(exc=exc))
        assert isinstance(out, QueryFailedResponse)
        assert out.error == QUERY_FAILED_MESSAGE
        assert "postgres" not in out.error and out.query == "q"


class TestRunGraphIsolation:
    async def test_repeated_same_query_does_not_accumulate(self, make_llm, fake_mcp: FakeNewsMcp) -> None:
        news = json.dumps({"is_news_query": True, "categories": ["technology"], "fetch_latest": False})
        llm = make_llm([news, "summary one", news, "summary two"])
        graph = await build(InMemorySaver(), llm, fake_mcp.client())
        first = await run_graph(NewsInput(query="tech news"), graph)
        second = await run_graph(NewsInput(query="tech news"), graph)
        assert isinstance(first, QuerySuccessResponse) and isinstance(second, QuerySuccessResponse)
        assert len(first.articles) == len(second.articles) == 1
