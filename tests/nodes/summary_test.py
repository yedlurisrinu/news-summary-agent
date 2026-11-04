"""
tests/nodes/summary_test.py

Unit tests for nodes/summary.py:
  - _build_context   — truncation to 1000 chars, missing date fallback, empty docs
  - Summary.__call__  — summary text + hits + timing
"""

from __future__ import annotations


class TestBuildContext:
    def test_empty_docs_returns_placeholder(self):
        from nodes.summary import Summary

        blocks, hits = Summary._build_context({"docs": []})
        assert blocks == ["No article found for summary"]
        assert hits == []

    def test_content_truncated_to_1000_chars(self, es_doc):
        from nodes.summary import Summary

        blocks, hits = Summary._build_context({"docs": [es_doc]})
        content_line = blocks[0].split("content: ", 1)[1]
        assert len(content_line) == 1000

    def test_article_hit_metadata(self, es_doc):
        from nodes.summary import Summary

        _, hits = Summary._build_context({"docs": [es_doc]})
        assert hits[0]["title"] == "AI Revolutionises Healthcare"
        assert hits[0]["published_at"] == "2026-03-28"

    def test_missing_published_at_falls_back(self):
        from nodes.summary import Summary

        doc = {
            "_source": {
                "title": "T",
                "source": "S",
                "link": "l",
                "content": "body",
            }
        }
        _, hits = Summary._build_context({"docs": [doc]})
        assert hits[0]["published_at"] == "date unknown"


class TestSummaryCall:
    def test_returns_summary_hits_and_timing(self, make_llm, make_es, es_doc):
        from nodes.summary import Summary

        es = make_es(mget_response={"docs": [es_doc]})
        node = Summary(make_llm(["Here is the grounded summary."]), es)

        result = node(
            {"query": "ai news", "article_refs": [{"article_id": "a1", "score": 1.0}]}
        )

        assert result["summary"] == "Here is the grounded summary."
        assert result["article_hits"][0]["title"] == "AI Revolutionises Healthcare"
        assert set(result["duration_seconds"]) == {"MGET_CAL", "SUMMARY_LLM_CALL"}
