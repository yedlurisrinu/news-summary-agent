from typing import TypedDict, Required


class SummaryFailed(TypedDict):
    """Error-shape channel (currently unused — see QueryFailedResponse)."""

    query: Required[str]
    error: Exception
    error_code: str
