"""Workflow entry point — invokes the compiled graph and maps state to API responses."""

import logging
from typing import Any
from uuid import uuid4

from fastapi import status
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from data.news_input import NewsInput
from data.query_failed_response import QueryFailedResponse
from data.query_success_response import QuerySuccessResponse
from exception.search_unavailable import SEARCH_UNAVAILABLE_MESSAGE, SearchUnavailable

logger = logging.getLogger(__name__)

QUERY_FAILED_MESSAGE = "Unable to process the query at this time, please try again later"


def cfg(tid: str, rl: int) -> RunnableConfig:
    """Build the RunnableConfig: thread_id for checkpointing + recursion limit."""
    return {"configurable": {"thread_id": tid}, "recursion_limit": rl}


async def run_graph(
    user_input: NewsInput, graph: CompiledStateGraph[Any, None, Any, Any]
) -> QuerySuccessResponse | QueryFailedResponse:
    """Invoke the compiled graph for one query and map the final state to a response.

    Each request uses a fresh uuid4 thread_id so every run is independent (a
    deterministic thread_id resumed old checkpoints and accumulated article_hits).
    Any exception is logged and returned as a ``QueryFailedResponse`` with fixed text
    only: "search unavailable" (503, retry later) for ``SearchUnavailable`` (an
    ``article_search`` failure), a generic message (500) otherwise. Other MCP failures do
    not get here: classify ignores a ``category_search`` failure, and a failed ``mget``
    returns 200 with fixed text from the summary node. Exception details never reach the
    client. The HTTP status of POST /query is set from ``error_code``.
    """
    try:
        out: dict[str, Any] = await graph.ainvoke(
            {"query": user_input.query}, cfg(uuid4().hex, 10)
        )
        logger.info("Total Time Taken = %s", out["duration_seconds"])
        response = QuerySuccessResponse(
            query=out["query"],
            categories=out["categories"],
            summary=out["summary"],
            articles=out["article_hits"],
            duration_seconds=sum(out["duration_seconds"].values()),
        )
    except Exception as e:
        logger.exception("Exception while agents are fetching news summary")
        if isinstance(e, SearchUnavailable):
            return QueryFailedResponse(
                query=user_input.query,
                error=SEARCH_UNAVAILABLE_MESSAGE,
                error_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return QueryFailedResponse(
            query=user_input.query,
            error=QUERY_FAILED_MESSAGE,
            error_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return response
