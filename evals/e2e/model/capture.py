"""A frozen e2e capture loaded from fixtures/e2e/<case_id>/<run_id>/."""

from dataclasses import dataclass
from typing import Any

from evals.e2e.model.article_hit import ArticleHit


@dataclass(frozen=True)
class Capture:
    """One frozen e2e case run: the HTTP response plus the query, summary and articles from the final state."""

    case_id: str
    run_id: str
    http_status: int
    query: str
    summary: str
    article_hits: tuple[ArticleHit, ...]
    response: dict[str, Any]
