"""
TODO in phase-1 I will be using multi match; embeddings, inference and sematic search will be completed later
Responsibility:
  Search the news_articles index using Elasticsearch semantic_text queries
  (inference handled with client-side embedding).

  When fetch_latest=True, results are filtered to the most recent 7 days
  and sorted by published_at descending before semantic re-ranking.

Technology: Elasticsearch Python client (semantic query), TODO tracing.
"""

from fastmcp import Client
from fastmcp.client.client import CallToolResult
from mcp_types import TextContent
from channels.search_results import SearchResult
from channels.news_state import NewsState
from fastmcp.client.transports import StreamableHttpTransport
import time
import logging

from data.news_query import NewsQuery
from exception.search_unavailable import SearchUnavailable

logger = logging.getLogger(__file__)


class Search:
    """Graph node: retrieve candidate article references from Elasticsearch."""

    def __init__(
        self,mcp_client: Client[StreamableHttpTransport]
    ):
        """Store the shared chat model and Elasticsearch client."""
        self._mcp_client = mcp_client

    async def __call__(self, state: NewsState) -> SearchResult:
        """Run the multi_match search (optionally filtered to the last 7 days) and return article refs + timing."""
        # Search es with query for the results
        # capture response time
        start = time.time()
        logger.debug(f"search intent incoming state : {state}")

        try:
            async with self._mcp_client:
                tool_result: CallToolResult = await self._mcp_client.call_tool(
                    "article_search",
                    {
                        "user_query": NewsQuery.model_validate(
                            {
                                "query": state["query"],
                                "categories": state["categories"],
                                "fetch_latest": state["fetch_latest"],
                            }
                        )
                    },
                )
                logger.info(f"Search result = {tool_result}")
            
            logger.info(
                    f"Response from Elasticsearch MCP Server : {tool_result.structured_content}"
                )
            sc = tool_result.structured_content
            if tool_result.is_error or sc is None:
                block = tool_result.content[0] if tool_result.content else None
                logger.error(block.text if isinstance(block, TextContent) else "mcp tool returned an error")
                raise SearchUnavailable(" Search is unavailable at this time, please try after few sec")
            
            return {
                "article_refs": sc["article_refs"],
                "duration_seconds": {"SEARCH": time.time() - start},
            }
        except Exception:
            logger.exception(
                "Exception while searching for articles for query: %s", state["query"]
            )
            raise

       
