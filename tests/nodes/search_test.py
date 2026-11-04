"""
tests/nodes/search_test.py

Unit tests for nodes/search.py:
  - _build_search_query   — multi_match query set, should dropped w/o categories, template not mutated
  - _append_latest_query  — 7-day range filter + recency-first sort
  - _get_article_refs     — (article_id, score) extraction
  - Search.__call__        — refs + timing, fetch_latest path
"""

from __future__ import annotations


class TestBuildSearchQuery:
    def test_sets_user_query_on_multi_match(self):
        from nodes.search import Search

        q = Search._build_search_query("ai news", ["technology"])
        assert q["query"]["bool"]["must"][0]["multi_match"]["query"] == "ai news"

    def test_categories_joined_into_should(self):
        from nodes.search import Search

        q = Search._build_search_query("ai news", ["technology", "ai"])
        assert (
            q["query"]["bool"]["should"][0]["multi_match"]["query"] == "technology ai"
        )

    def test_should_dropped_when_no_categories(self):
        from nodes.search import Search

        q = Search._build_search_query("ai news", [])
        assert "should" not in q["query"]["bool"]

    def test_does_not_mutate_template(self):
        from config.application_config import get_config
        from nodes.search import Search

        Search._build_search_query("ai news", ["technology"])
        tmpl = get_config("SEARCH_QUERY_TEMPLATE")
        assert tmpl["query"]["bool"]["must"][0]["multi_match"]["query"] == ""


class TestAppendLatestQuery:
    def test_adds_range_filter_and_sort(self):
        from nodes.search import Search

        base = {"query": {"bool": {}}}
        out = Search._append_latest_query(base)
        assert "published_at" in out["query"]["bool"]["filter"][0]["range"]
        assert out["sort"][0]["published_at"]["order"] == "desc"


class TestGetArticleRefs:
    def test_parses_hits(self, es_hit):
        from nodes.search import Search

        resp = {"hits": {"hits": [es_hit]}}
        refs = Search._get_article_refs(resp)
        assert refs == [{"article_id": "a1", "score": 1.23}]

    def test_empty_hits(self):
        from nodes.search import Search

        assert Search._get_article_refs({"hits": {"hits": []}}) == []


class TestSearchCall:
    def test_returns_refs_and_timing(self, make_llm, make_es, es_hit):
        from nodes.search import Search

        es = make_es(search_response={"hits": {"hits": [es_hit]}})
        node = Search(make_llm([""]), es)

        result = node(
            {"query": "ai news", "categories": ["technology"], "fetch_latest": False}
        )

        assert result["article_refs"] == [{"article_id": "a1", "score": 1.23}]
        assert "SEARCH" in result["duration_seconds"]

    def test_fetch_latest_applies_filter(self, make_llm, make_es, es_hit):
        from nodes.search import Search

        es = make_es(search_response={"hits": {"hits": [es_hit]}})
        node = Search(make_llm([""]), es)

        node({"query": "ai news", "categories": ["technology"], "fetch_latest": True})

        sent_query = es.search.call_args.kwargs
        assert "filter" in sent_query["query"]["bool"]

    def test_es_exception_propagates(self, make_llm, make_es):
        import pytest

        from nodes.search import Search

        es = make_es()
        es.search.side_effect = RuntimeError("ES down")
        node = Search(make_llm([""]), es)

        with pytest.raises(RuntimeError):
            node({"query": "ai news", "categories": [], "fetch_latest": False})
