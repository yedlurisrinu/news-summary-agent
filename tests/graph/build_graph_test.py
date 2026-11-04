"""
tests/graph/build_graph_test.py

Tests for graph/build_graph.py:
  - route_after_classify / route_after_search — routing decisions
  - end-to-end routing through a compiled graph (MemorySaver + fakes) for the
    three cases: normal summary, off-topic guard rail, no-results guard rail.
"""

from __future__ import annotations

NEWS_TECH = '{"is_news_query": true, "categories": ["ai"], "fetch_latest": false}'
OFF_TOPIC = '{"is_news_query": false, "categories": [], "fetch_latest": false}'
SUMMARY_PROSE = "Grounded summary of the retrieved articles."


class TestRouters:
    def test_route_after_classify_news(self):
        from graph.build_graph import route_after_classify

        assert route_after_classify({"is_news_query": True}) == "search"

    def test_route_after_classify_off_topic(self):
        from graph.build_graph import route_after_classify

        assert route_after_classify({"is_news_query": False}) == "reply_off_topic"

    def test_route_after_search_found(self):
        from graph.build_graph import route_after_search

        assert (
            route_after_search({"article_refs": [{"article_id": "a1", "score": 1.0}]})
            == "found"
        )

    def test_route_after_search_empty(self):
        from graph.build_graph import route_after_search

        assert route_after_search({"article_refs": []}) == "empty"


def _compile(llm, es):
    from langgraph.checkpoint.memory import MemorySaver

    from graph.build_graph import build

    return build(MemorySaver(), llm, es)


CFG = {"configurable": {"thread_id": "t-test"}}


class TestEndToEndRouting:
    def test_normal_path_produces_summary(self, make_llm, make_es, es_hit, es_doc):
        graph = _compile(
            make_llm([NEWS_TECH, SUMMARY_PROSE]),
            make_es(
                search_response={"hits": {"total": {"value": 3}, "hits": [es_hit]}},
                mget_response={"docs": [es_doc]},
            ),
        )
        out = graph.invoke({"query": "ai news"}, CFG)
        assert out["summary"] == SUMMARY_PROSE
        assert out["article_hits"]

    def test_off_topic_hits_guard_rail_1(self, make_llm, make_es):
        from data.seed_data import SeedData

        graph = _compile(make_llm([OFF_TOPIC]), make_es())
        out = graph.invoke({"query": "write me a poem"}, CFG)
        assert out["summary"] == SeedData._OFF_TOPIC_REPLY

    def test_no_results_hits_guard_rail_2(self, make_llm, make_es):
        from data.seed_data import SeedData

        graph = _compile(
            make_llm([NEWS_TECH]),
            make_es(search_response={"hits": {"total": {"value": 0}, "hits": []}}),
        )
        out = graph.invoke({"query": "obscure topic news"}, CFG)
        assert out["summary"] == SeedData._NO_NEWS_FOUND_REPLY
