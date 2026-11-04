"""
tests/graph/run_graph_workflow_test.py

Tests for graph/run_graph_workflow.py:
  - cfg()       — RunnableConfig shape
  - run_graph() — success mapping to QuerySuccessResponse, exception → QueryFailedResponse(500)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

STATE = {
    "query": "ai news",
    "categories": ["ai"],
    "summary": "A grounded summary.",
    "article_hits": [
        {
            "title": "AI Healthcare",
            "source": "TechNews",
            "published_at": "2026-03-28",
            "link": "https://example.com/ai",
        }
    ],
    "duration_seconds": {"CLASSIFY": 0.1, "SEARCH": 0.2},
}


class TestCfg:
    def test_shape(self):
        from graph.run_graph_workflow import cfg

        assert cfg("tid-1", 10) == {
            "configurable": {"thread_id": "tid-1"},
            "recursion_limit": 10,
        }


class TestRunGraph:
    def test_success_maps_state_to_response(self):
        from data.news_input import NewsInput
        from graph.run_graph_workflow import run_graph

        graph = MagicMock()
        graph.invoke.return_value = STATE

        resp = run_graph(NewsInput(query="ai news"), graph)

        assert resp.status == "ok"
        assert resp.summary == "A grounded summary."
        assert resp.categories == ["ai"]
        assert resp.duration_seconds == pytest.approx(0.3)
        assert resp.articles[0].title == "AI Healthcare"

    def test_exception_returns_failed_response(self):
        from data.news_input import NewsInput
        from graph.run_graph_workflow import run_graph

        graph = MagicMock()
        graph.invoke.side_effect = RuntimeError("boom")

        resp = run_graph(NewsInput(query="ai news"), graph)

        assert resp.status == "failed"
        assert resp.error_code == 500
        assert "boom" in resp.error


class TestRunGraphIsolation:
    def test_repeated_same_query_does_not_accumulate(
        self, make_llm, make_es, es_hit, es_doc
    ):
        """Regression: a fresh uuid4 thread_id per request keeps runs independent.

        Under the old sha256(query) thread_id, the second identical query resumed
        the persisted checkpoint and merge_article_hits appended, so article_hits
        grew (1 → 2). A unique thread_id keeps every run at the true count.
        """
        from langgraph.checkpoint.memory import MemorySaver

        from data.news_input import NewsInput
        from graph.build_graph import build
        from graph.run_graph_workflow import run_graph

        llm = make_llm(
            [
                '{"is_news_query": true, "categories": ["ai"], "fetch_latest": false}',
                "summary",
            ]
            * 3
        )
        es = make_es(
            search_response={"hits": {"total": {"value": 3}, "hits": [es_hit]}},
            mget_response={"docs": [es_doc]},
        )
        graph = build(MemorySaver(), llm, es)

        first = run_graph(NewsInput(query="ai news"), graph)
        second = run_graph(NewsInput(query="ai news"), graph)

        assert first.status == "ok" and second.status == "ok"
        assert len(first.articles) == 1
        assert (
            len(second.articles) == 1
        )  # would be 2 with the old deterministic thread_id
