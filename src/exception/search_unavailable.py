SEARCH_UNAVAILABLE_MESSAGE = (
    "Search is unavailable at this time, please try again in a few seconds"
)


class SearchUnavailable(Exception):
    """An MCP search tool failed or returned an error; the client only ever sees ``SEARCH_UNAVAILABLE_MESSAGE``."""

    def __init__(self, message: str = SEARCH_UNAVAILABLE_MESSAGE) -> None:
        super().__init__(message)
