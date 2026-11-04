from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Inbound POST /query request body."""

    query: str = Field(..., description="Natural-language news query to summarize.")
