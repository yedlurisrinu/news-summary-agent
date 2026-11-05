from typing import Required, TypedDict

from channels.article_hit import ArticleHit
from channels.article_reference import ArticleReference


class SummaryOk(TypedDict):
    """Graph output schema for a successful run (full enriched state)."""

    query: Required[str]
    is_news_query: Required[bool]
    fetch_latest: Required[bool]
    categories: Required[list[str]]
    article_refs: Required[list[ArticleReference]]
    article_hits: Required[list[ArticleHit]]
    summary: Required[str]
    duration_seconds: Required[dict[str, float]]
