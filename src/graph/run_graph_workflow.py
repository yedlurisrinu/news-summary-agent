"""Workflow entry point — invokes the compiled graph and maps state to API responses."""

from uuid import uuid4

# from hashlib import sha256  # retained for reference; see the commented thread_id logic in run_graph
from typing import Any
from langchain_core.runnables import RunnableConfig
from data.news_input import NewsInput
from data.query_failed_response import QueryFailedResponse
from data.query_success_response import QuerySuccessResponse
from langgraph.graph.state import CompiledStateGraph
from fastapi import status
import logging

logger = logging.getLogger(__file__)


def cfg(tid: str, rl: int) -> RunnableConfig:
    """Build the RunnableConfig: thread_id for checkpointing + recursion limit."""
    return {"configurable": {"thread_id": tid}, "recursion_limit": rl}


async def run_graph(
    user_input: NewsInput, graph: CompiledStateGraph[Any, None, Any, Any]
) -> QuerySuccessResponse | QueryFailedResponse:
    """Invoke the compiled graph for one query and map the final state to a response.

    Each request uses a fresh uuid4 thread_id so every run is independent. Any
    exception is caught and returned as a ``QueryFailedResponse`` (500).
    """
    try:
        # Previous approach — deterministic thread_id = sha256 of the normalized query:
        #   out = graph.invoke({"query": user_input.query},
        #                      cfg(sha256(user_input.query.strip().casefold().encode()).hexdigest(), 10))
        # This made article_hits accumulate across repeated identical queries: the persisted
        # checkpoint was resumed and re-run, and merge_article_hits appends without dedupe.
        out: dict[str, Any] = await graph.ainvoke(
            {"query": user_input.query}, cfg(uuid4().hex, 10)
        )
        logging.info(f"Total Time Taken = {out['duration_seconds']}")
        response = QuerySuccessResponse(
            query=out["query"],
            categories=out["categories"],
            summary=out["summary"],
            articles=out["article_hits"],
            duration_seconds=sum(out["duration_seconds"].values()),
        )
    except Exception as e:
        logger.exception(f" Exception while agents are fetching news summary: {e}")
        return QueryFailedResponse(
            query=user_input.query,
            error=str(e) or type(e).__name__,
            error_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return response
