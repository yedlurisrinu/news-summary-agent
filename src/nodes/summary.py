"""Summary node — mget full article bodies and produce a grounded, cited LLM summary."""

from fastmcp import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastmcp.client.client import CallToolResult
from mcp_types import TextContent

from channels.article_hit import ArticleHit
from channels.summary_result import SummaryResult
from channels.news_state import NewsState
from langchain_core.language_models import BaseChatModel
from elasticsearch import AsyncElasticsearch
import time
import logging
from fastmcp.client.transports import StreamableHttpTransport

logger = logging.getLogger(__file__)

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


class Summary:
    """Graph node: fetch article content by id and generate a cited, grounded summary."""

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

    async def __call__(self, state: NewsState) -> SummaryResult:
        """mget article bodies, build context, invoke the summarization chain; returns summary + hits + timing."""
        logger.debug(f"summary intent incoming state : {state}")
        duration_seconds: dict[str, float] = {}
        # Get actual content
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
                sc = tool_result.structured_content
                if tool_result.is_error or sc is None:
                    block = tool_result.content[0] if tool_result.content is not None else None
                    msg = block.text if isinstance(block, TextContent) else "mcp search tool failed"
                    logger.error(msg)
                    (duration_seconds.setdefault("MGET_CAL", time.time() - MGET_START),)
                    return {
                        "summary": "Unable to fetch articles at this time",
                        "article_hits": [],
                        "duration_seconds": duration_seconds,
                    }
                
            duration_seconds.setdefault("MGET_CAL", time.time() - MGET_START)
            # build context
            SUMMARY_CALL = time.time()
            context: list[str] = []
            article_hits: list[ArticleHit] = sc["article_hits"]
            context = await Summary._build_context(article_hits)
            # call model with news articles
            chain = _SUMMARY_PROMPT | self._llm | StrOutputParser()
            # LLM call for summarization
            summary: str = await chain.ainvoke(
                {"query": state["query"], "context": context}
            )
            # capture response time
            duration_seconds.setdefault("SUMMARY_LLM_CALL", time.time() - SUMMARY_CALL)
            return {
                "summary": summary,
                "article_hits": article_hits,
                "duration_seconds": duration_seconds,
            }
        except Exception as e:
            logger.exception(e)
            (duration_seconds.setdefault("MGET_CAL", time.time() - MGET_START),)
            return {
                "summary": "Unable to fetch articles at this time",
                "article_hits": [],
                "duration_seconds": duration_seconds,
            }

    @staticmethod
    async def _build_context(
        hits: list[ArticleHit],
    ) -> list[str]:
        """Turn an ES mget response into LLM context blocks and ArticleHit metadata (content truncated to 1000 chars)."""
        blocks: list[str] = []
        for i, hit in enumerate(hits):
            blocks.append(
                f"Article {i}: \n"
                f"title: {hit.get('title')}\n"
                f"source:{hit.get('source')}\n"
                f"published at: {hit.get('published_at')}\n"
                f"content: {hit.get('content', '')}"
            )

        return blocks
