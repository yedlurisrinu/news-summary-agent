"""Graph assembly — wires classify → search → summarize with two guard-rail routers."""

import logging

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from adapters.mcp_session import McpToolClient
from channels.news_state import NewsState
from channels.summary_ok import SummaryOk
from channels.summary_result import SummaryResult
from data.news_input import NewsInput
from data.seed_data import SeedData
from nodes.classification import ClassifyIntent
from nodes.search import Search
from nodes.summary import Summary

logger = logging.getLogger(__name__)
from typing import Any, Literal


async def reply_off_topic(state: NewsState) -> SummaryResult:
    """
    User is not allowed to send query that is off the allowed topics, this is the response agents will give
    Guard Rail #1
    """
    return {
        "summary": SeedData._OFF_TOPIC_REPLY,
        "article_hits": [],
        "duration_seconds": {},
    }


async def reply_no_results(state: NewsState) -> SummaryResult:
    """
    User requested query criteria did not yield any news articles to summarize
    Guard Rail #2
    """
    return {
        "summary": SeedData._NO_NEWS_FOUND_REPLY,
        "article_hits": [],
        "duration_seconds": {},
    }


async def route_after_classify(
    state: NewsState,
) -> Literal["search", "reply_off_topic"]:
    """Router to route to guard rail#1 node or search node - making sure non news related queries are not honored"""
    return "search" if state["is_news_query"] else "reply_off_topic"


async def route_after_search(state: NewsState) -> Literal["found", "empty"]:
    """Router to route to guard rail#2 node or summary node - making sure an empty search result does not trigger an LLM call"""
    return "found" if state["article_refs"] else "empty"


async def build(
    saver: AsyncPostgresSaver,
    llm: BaseChatModel,
    mcp_client: McpToolClient,
) -> CompiledStateGraph[NewsState, None, NewsInput, SummaryOk]:
    """
    Building state graph workflow with nodes classify, search, summary, off topic (Guard #1), no result (Guard#2)
    """
    g = StateGraph(NewsState, input_schema=NewsInput, output_schema=SummaryOk)
    nodes: list[tuple[str, Any]] = [
        ("classify", ClassifyIntent(llm, mcp_client)),
        ("search", Search(mcp_client)),
        ("summarize", Summary(llm, mcp_client)),
        ("reply_off_topic", reply_off_topic),
        ("reply_no_results", reply_no_results),
    ]
    for name, fn in nodes:
        g.add_node(name, fn)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify", route_after_classify, ["search", "reply_off_topic"]
    )
    g.add_conditional_edges(
        "search",
        route_after_search,
        {"found": "summarize", "empty": "reply_no_results"},
    )
    for terminal in ("summarize", "reply_off_topic", "reply_no_results"):
        g.add_edge(terminal, END)
    return g.compile(checkpointer=saver)
