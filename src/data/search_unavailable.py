from pydantic import BaseModel
from typing import Annotated
from pydantic.fields import Field

class NoArticleFound(BaseModel):
    message: Annotated[str, Field(description=" actual reason if available otherwise a generic message ")] = "No articles found for the search criteria"