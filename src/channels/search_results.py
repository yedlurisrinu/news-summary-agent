from typing import Required, TypedDict

from channels.article_reference import ArticleReference


class SearchResult(TypedDict):
    """Output of Search node"""

    article_refs: Required[list[ArticleReference]]
    duration_seconds: Required[dict[str, float]]  # Time took
