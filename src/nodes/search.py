"""Search node — find candidate articles through es-mcp-server's ``article_search`` MCP tool.

The tool runs a multi_match query on title and content (boosted by the classified
categories); with ``fetch_latest`` it keeps the last 7 days, newest first. The node
returns article references (id + score); content is fetched later by the summary node.
"""

import logging
import time

from fastmcp.client.client import CallToolResult
from mcp_types import TextContent

from adapters.mcp_session import McpToolClient
from channels.news_state import NewsState
from channels.search_results import SearchResult
from data.news_query import NewsQuery
from exception.search_unavailable import SearchUnavailable

logger = logging.getLogger(__name__)


class Search:
    """Graph node: retrieve candidate article references through the ``article_search`` MCP tool."""

    def __init__(self, mcp_client: McpToolClient):
        """Store the shared MCP client."""
        self._mcp_client = mcp_client

    async def __call__(self, state: NewsState) -> SearchResult:
        """Call ``article_search`` (optionally limited to the last 7 days) and return article refs + timing.

        Any tool error or transport failure becomes ``SearchUnavailable`` with fixed text.
        """
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
                    raise_on_error=False,
                )
        except Exception as exc:
            logger.exception("article_search call failed for query: %s", state["query"])
            raise SearchUnavailable() from exc

        sc = tool_result.structured_content
        if tool_result.is_error or sc is None:
            block = tool_result.content[0] if tool_result.content else None
            logger.error(
                "article_search returned an error: %r",
                block.text if isinstance(block, TextContent) else "no text content",
            )
            raise SearchUnavailable()

        logger.info("article_search: %d refs", len(sc["article_refs"]))
        return {
            "article_refs": sc["article_refs"],
            "duration_seconds": {"SEARCH": time.time() - start},
        }
