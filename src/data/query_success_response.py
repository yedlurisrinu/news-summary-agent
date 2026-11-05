from typing import Literal

from pydantic import BaseModel, Field


class ArticleOut(BaseModel):
    """One article referenced in the API response."""

    title: str = Field(..., description="Article headline.")
    link: str = Field(..., description="Canonical URL of the article.")
    source: str = Field(..., description="Publisher / source name.")
    published_at: str = Field(
        ..., description="Publication date (YYYY-MM-DD) or 'date unknown'."
    )


class QuerySuccessResponse(BaseModel):
    """Successful /query response (status='ok')."""

    query: str = Field(..., description="The original user query.")
    categories: list[str] = Field(
        ..., description="News categories detected in the query."
    )
    summary: str = Field(
        ..., description="Grounded, cited summary of the retrieved articles."
    )
    articles: list[ArticleOut] = Field(
        ..., description="Articles used to build the summary."
    )
    duration_seconds: float = Field(
        ..., description="Total wall-clock time across all graph stages."
    )
    status: Literal["ok"] = "ok"
