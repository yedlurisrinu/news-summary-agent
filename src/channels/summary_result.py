from typing import Required, TypedDict

from channels.article_hit import ArticleHit


class SummaryResult(TypedDict):
    """Output of the Summary node (and the guard-rail reply nodes)."""

    summary: Required[str]
    article_hits: Required[list[ArticleHit]]
    duration_seconds: Required[dict[str, float]]  # Time this agent took
