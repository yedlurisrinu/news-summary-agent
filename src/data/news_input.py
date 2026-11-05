from pydantic import BaseModel, Field


class NewsInput(BaseModel):
    """Internal request object passed into the graph workflow."""

    query: str = Field(..., description="Natural-language news query from the user.")
