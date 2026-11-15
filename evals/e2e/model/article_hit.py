"""The article as the judge sees it (mget output from the captured state)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ArticleHit:
    """A single retrieved news article."""

    article_id: str
    title: str
    source: str
    published_at: str
    link: str
    content: str
