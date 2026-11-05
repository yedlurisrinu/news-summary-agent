"""
tests/data/responses_test.py

Tests for the Pydantic response models in data/:
  - QuerySuccessResponse / ArticleOut — default status, field coercion
  - QueryFailedResponse               — default status, int error_code
"""

from __future__ import annotations


class TestQuerySuccessResponse:
    def test_defaults_and_nested_article(self):
        from data.query_success_response import QuerySuccessResponse

        resp = QuerySuccessResponse(
            query="ai news",
            categories=["ai"],
            summary="A summary.",
            articles=[
                {
                    "title": "T",
                    "link": "https://x",
                    "source": "S",
                    "published_at": "2026-01-01",
                }
            ],
            duration_seconds=0.5,
        )
        assert resp.status == "ok"
        assert resp.articles[0].title == "T"


class TestQueryFailedResponse:
    def test_defaults(self):
        from data.query_failed_response import QueryFailedResponse

        resp = QueryFailedResponse(query="ai news", error="boom", error_code=500)
        assert resp.status == "failed"
        assert resp.error_code == 500
