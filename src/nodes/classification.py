"""Classification node — detect news intent and categories from the user query."""

import copy
from typing import Any, cast

from elasticsearch import AsyncElasticsearch
from fastmcp import Client
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastmcp.client.client import CallToolResult
from mcp_types import TextContent
from channels.news_state import NewsState
from channels.category_search_results import CategorySearchResult
from langchain_core.language_models import BaseChatModel
from fastmcp.client.transports import StreamableHttpTransport
import time
import logging

logger = logging.getLogger(__file__)

_CATEGORY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a news query analyser. Given a user query, extract:
1. `is_news_query`: true if the query is about news, current events, or any topic
   that could appear in a news feed (politics, technology, sports, business, science,
   health, climate, finance, entertainment, world affairs, etc.).
   Set to false for general knowledge questions, coding help, math, creative writing,
   personal advice, opinions, or anything clearly unrelated to news.
2. `categories`: the news topic categories being asked about.
   Use well-known category names such as: technology, tech, politics, sports, business,
   entertainment, science, health, world, finance, fin, climate, ai, crypto.
   If the query spans multiple topics return all of them.
   Return an empty list when is_news_query is false.
3. `fetch_latest`: true only when the user explicitly wants recent/latest/today's news.
   Always false when is_news_query is false.

Respond ONLY with valid JSON matching this schema:
{{"is_news_query": true/false, "categories": ["..."], "fetch_latest": true/false}}""",
        ),
        ("human", "{query}"),
    ]
)


class ClassifyIntent:
    """Graph node: classify a user query's news intent and categories.

    Responsibilities:
      1. Parse the query to identify news categories of interest.
      2. Verify those categories exist in the news_articles index.
      3. Detect recency intent (``fetch_latest``).
      4. Return a :class:`CategorySearchResult`.

    Tech: LangChain LCEL + OpenAI + a direct Elasticsearch terms aggregation.
    Tracing: TODO OpenTelemetry (Jaeger) integration.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        es: AsyncElasticsearch,
        mcp_client: Client[StreamableHttpTransport],
    ):
        """Store the shared chat model and Elasticsearch client."""
        self._llm = llm
        self._es = es
        self._mcp_client = mcp_client

    async def __call__(self, state: NewsState) -> CategorySearchResult:
        """Classify intent; returns early (no ES call) for non-news queries."""
        # make intend call to model and get the response
        logger.debug(f"classify intent incoming state : {state}")

        duration_seconds: dict[str, float] = {}
        llm_start = time.time()
        chain = _CATEGORY_PROMPT | self._llm | JsonOutputParser()
        result = await chain.ainvoke({"query": state["query"]})
        duration_seconds.setdefault("LLM_LATENCY", time.time() - llm_start)
        # check if query for latest news
        if result.get("is_news_query") and result.get("categories"):
            # make call to es for categories
            es_call_start = time.time()
            found = await self._resolve_categories(result.get("categories"))
            duration_seconds.setdefault("ES_CALL", time.time() - es_call_start)
            result["duration_seconds"] = duration_seconds
            logger.info(
                f"ClassifyIntent: Categories={result.get('categories')}, Total Found={found}, fetch_latest={result.get('fetch_latest')}, is_news_query={result.get('is_news_query')}"
            )

        return cast(CategorySearchResult, result)

    async def _resolve_categories(self, categories: list[str]) -> int:
        """
        Run a terms aggregation on source.keyword to confirm at least one article
        exists for each requested category.  Falls back to returning the LLM
        categories as-is if ES has no 'category' field — the search agent will
        still use them as semantic query hints.
        """
        try:
            async with self._mcp_client:
                tool_result: CallToolResult = await self._mcp_client.call_tool(
                    "category_search", {"categories": categories}, raise_on_error=False
                )
                sc = tool_result.structured_content
                if tool_result.is_error or sc is None:
                    block = tool_result.content[0] if tool_result.content else None
                    logger.error(block.text if isinstance(block, TextContent) else "mcp tool returned an error")
                    return 0
                
                total_hits: int = sc["result"]
                logger.info(f"Classify result = {total_hits}")

            if total_hits == 0:
                logger.warning(
                    "No ES documents matched categories %s — proceeding anyway.",
                    categories,
                )
            else:
                logger.info(
                    "Category check: %d documents matched for %s",
                    total_hits,
                    categories,
                )
            return total_hits
        except Exception:  # noqa: BLE001
            logger.exception("ES category verification failed skipping check.")

        return 0
