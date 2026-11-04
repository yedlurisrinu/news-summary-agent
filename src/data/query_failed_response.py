from typing import Literal

from pydantic import BaseModel, Field


class QueryFailedResponse(BaseModel):
    """Failed /query response (status='failed')."""

    query: str = Field(..., description="The original user query.")
    error: str = Field(
        ..., description="Human-readable error message or exception type."
    )
    error_code: int = Field(..., description="HTTP-style status code for the failure.")
    status: Literal["failed"] = "failed"
