from typing import Required, TypedDict


class CategorySearchResult(TypedDict):
    """Output of the ClassifyIntent node.

    categories: news category names extracted from the query (lowercase, singular
        nouns, e.g. 'technology', 'politics', 'sports'); empty when is_news_query is False.
    fetch_latest: True when the user explicitly asks for latest/recent/today's/newest articles.
    is_news_query: True for news/current-events/real-world-topic queries; False for
        general knowledge, opinions, creative writing, coding, math, or personal advice.
    duration_seconds: time spent at key external calls within this node.
    """

    categories: list[str]
    fetch_latest: Required[bool]
    is_news_query: Required[bool]
    duration_seconds: Required[dict[str, float]]
