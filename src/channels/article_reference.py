from typing import Required, TypedDict


class ArticleReference(TypedDict):
    """A scored reference to an article (id + relevance score), pre-summary."""

    article_id: Required[str]
    score: Required[float]
