from typing import Required, TypedDict

from channels.article_reference import ArticleReference
from channels.latencies import Latency


class SearchResult(TypedDict):
    """Output of Search node"""

    article_refs: Required[list[ArticleReference]]
    latency: Required[Latency]
