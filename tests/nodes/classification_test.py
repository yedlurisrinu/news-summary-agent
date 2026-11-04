"""
tests/nodes/classification_test.py

Unit tests for nodes/classification.py:
  - category_query()        — bool-should clause per category, template not mutated
  - ClassifyIntent.__call__ — non-news short-circuit (no ES call), news-query path
"""

from __future__ import annotations


class TestCategoryQuery:
    def test_one_should_clause_per_category(self):
        from nodes.classification import category_query

        q = category_query(["technology", "ai"])
        should = q["query"]["bool"]["should"]
        titles = [c["match"]["title"] for c in should]
        assert titles == ["technology", "ai"]

    def test_does_not_mutate_loaded_template(self):
        from config.application_config import get_config
        from nodes.classification import category_query

        category_query(["technology"])
        assert get_config("CATEGORY_QUERY")["query"]["bool"]["should"] == []


class TestClassifyIntentCall:
    def test_non_news_query_short_circuits(self, make_llm, make_es):
        from nodes.classification import ClassifyIntent

        llm = make_llm(
            ['{"is_news_query": false, "categories": [], "fetch_latest": false}']
        )
        es = make_es()
        node = ClassifyIntent(llm, es)

        result = node({"query": "write me a poem"})

        assert result["is_news_query"] is False
        assert result["duration_seconds"].keys() == {"CLASSIFY_LLM_LATENCY"}
        es.search.assert_not_called()

    def test_news_query_resolves_categories_and_times(self, make_llm, make_es):
        from nodes.classification import ClassifyIntent

        llm = make_llm(
            [
                '{"is_news_query": true, "categories": ["technology"], "fetch_latest": true}'
            ]
        )
        es = make_es(search_response={"hits": {"total": {"value": 3}, "hits": []}})
        node = ClassifyIntent(llm, es)

        result = node({"query": "latest tech news"})

        assert result["is_news_query"] is True
        assert result["categories"] == ["technology"]
        assert result["fetch_latest"] is True
        assert set(result["duration_seconds"]) == {"LLM_LATENCY", "ES_CALL"}
        es.search.assert_called_once()

    def test_es_failure_is_swallowed(self, make_llm, make_es):
        from nodes.classification import ClassifyIntent

        llm = make_llm(
            ['{"is_news_query": true, "categories": ["sports"], "fetch_latest": false}']
        )
        es = make_es()
        es.search.side_effect = RuntimeError("ES down")
        node = ClassifyIntent(llm, es)

        # _resolve_categories catches the exception and returns 0 — call still succeeds.
        result = node({"query": "sports news"})
        assert result["is_news_query"] is True
