from typing import Required, TypedDict

from channels.article_hit import ArticleHit
from channels.latencies import Latency


class SummaryResult(TypedDict):
    """Output of the Summary node (and the guard-rail reply nodes)."""

    summary: Required[str]
    article_hits: Required[list[ArticleHit]]
    latency: Required[Latency]
