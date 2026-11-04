from typing import TypedDict, Required


class ArticleReference(TypedDict):
    """A scored reference to an article (id + relevance score), pre-summary."""

    article_id: Required[str]
    score: Required[float]
