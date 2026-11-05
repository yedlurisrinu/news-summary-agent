"""Summary node — mget full article bodies and produce a grounded, cited LLM summary."""

import logging
import time

from fastmcp.client.client import CallToolResult
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from mcp_types import TextContent

from adapters.mcp_session import McpToolClient
from channels.article_hit import ArticleHit
from channels.news_state import NewsState
from channels.summary_result import SummaryResult
from config.application_config import get_config
from exception.search_unavailable import SEARCH_UNAVAILABLE_MESSAGE

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert news analyst. You will receive a user query and a
collection of news article snippets retrieved from a search engine.

Your task:
- Write a clear, factual summary of 3–5 paragraphs that directly answers the query.
- Lead with the most important finding.
- When referencing a specific fact, cite the article title in brackets, e.g. [Article Title].
- Remain objective and neutral in tone.

Strict grounding rules — you MUST follow these without exception:
- Use ONLY information explicitly stated in the retrieved articles below.
- Do NOT use your training knowledge, general world knowledge, or any information
  not present in the provided context.
- Do NOT speculate, infer, or fill in gaps beyond what the articles state.
- If the provided articles do not contain enough information to answer the query,
  say so explicitly: state what was found and what remains unanswered.""",
        ),
        (
            "human",
            """USER QUERY: {query} RETRIEVED ARTICLES: {context}
Write the summary:""",
        ),
    ]
)


SUMMARY_FAILED_MESSAGE = "Unable to summarise the articles at this time"


class Summary:
    """Graph node: fetch article content by id and generate a cited, grounded summary."""

    def __init__(
        self,
        llm: BaseChatModel,
        mcp_client: McpToolClient,
    ):
        """Store the shared chat model and MCP client."""
        self._llm = llm
        self._mcp_client = mcp_client

    async def __call__(self, state: NewsState) -> SummaryResult:
        """mget article bodies, build context, invoke the summarization chain; returns summary + hits + timing.

        A failed ``mget`` replies with the fixed "search unavailable" text; a failed LLM
        call replies with ``SUMMARY_FAILED_MESSAGE``. Details are only logged.
        """
        logger.debug(f"summary intent incoming state : {state}")
        duration_seconds: dict[str, float] = {}
        MGET_START = time.time()
        try:
            async with self._mcp_client:
                tool_result: CallToolResult = await self._mcp_client.call_tool(
                    "mget",
                    {
                        "ids": [
                            article["article_id"] for article in state["article_refs"]
                        ]
                    },
                    raise_on_error=False,
                )
        except Exception:
            logger.exception("mget call failed")
            return Summary._reply(SEARCH_UNAVAILABLE_MESSAGE, duration_seconds, "MGET_CAL", MGET_START)

        sc = tool_result.structured_content
        if tool_result.is_error or sc is None:
            block = tool_result.content[0] if tool_result.content else None
            logger.error(
                "mget returned an error: %r",
                block.text if isinstance(block, TextContent) else "no text content",
            )
            return Summary._reply(SEARCH_UNAVAILABLE_MESSAGE, duration_seconds, "MGET_CAL", MGET_START)

        duration_seconds.setdefault("MGET_CAL", time.time() - MGET_START)
        SUMMARY_CALL = time.time()
        article_hits: list[ArticleHit] = sc["article_hits"]
        context = await Summary._build_context(article_hits)
        chain = _SUMMARY_PROMPT | self._llm | StrOutputParser()
        try:
            summary: str = await chain.ainvoke(
                {"query": state["query"], "context": context}
            )
        except Exception:
            logger.exception("summary LLM call failed")
            return Summary._reply(SUMMARY_FAILED_MESSAGE, duration_seconds, "SUMMARY_LLM_CALL", SUMMARY_CALL)
        duration_seconds.setdefault("SUMMARY_LLM_CALL", time.time() - SUMMARY_CALL)
        return {
            "summary": summary,
            "article_hits": article_hits,
            "duration_seconds": duration_seconds,
        }

    @staticmethod
    def _reply(
        text: str, duration_seconds: dict[str, float], step: str, started: float
    ) -> SummaryResult:
        """A fixed-text reply with no articles, recording how long the failed step took."""
        duration_seconds.setdefault(step, time.time() - started)
        return {"summary": text, "article_hits": [], "duration_seconds": duration_seconds}

    @staticmethod
    async def _build_context(
        hits: list[ArticleHit],
    ) -> list[str]:
        """Turn mget article hits into LLM context blocks, content capped at ``SUMMARY_CONTENT_MAX_CHARS``."""
        max_chars = int(get_config("SUMMARY_CONTENT_MAX_CHARS"))
        blocks: list[str] = []
        for i, hit in enumerate(hits):
            blocks.append(
                f"Article {i}: \n"
                f"title: {hit.get('title')}\n"
                f"source:{hit.get('source')}\n"
                f"published at: {hit.get('published_at')}\n"
                f"content: {str(hit.get('content') or '')[:max_chars]}"
            )

        return blocks
