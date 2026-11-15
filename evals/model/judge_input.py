"""One (case, criterion) item sent to the judge."""

from dataclasses import dataclass

from evals.model.article_hit import ArticleHit
from evals.model.models import Criterion


@dataclass(frozen=True)
class JudgeInput:
    """One item for the judge: a case's query, summary and articles, graded on one criterion."""

    case_id: str
    query: str
    summary: str
    articles: tuple[ArticleHit, ...]
    criterion: Criterion
