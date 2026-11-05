"""
tests/nodes/summary_test.py

Summary node with a fake LLM and a fake es-mcp-server:
  - context blocks: article metadata, content capped at SUMMARY_CONTENT_MAX_CHARS (1000)
  - success: ``mget`` by the state's ids, LLM summary, hits and timings
  - any mget failure replies with the fixed "search unavailable" text and no articles
  - an LLM failure replies with SUMMARY_FAILED_MESSAGE
"""

from __future__ import annotations

import pytest
from fastmcp.client.client import CallToolResult
from langchain_core.runnables import RunnableLambda
from mcp_types import ImageContent, TextContent

from exception.search_unavailable import SEARCH_UNAVAILABLE_MESSAGE
from nodes.summary import SUMMARY_FAILED_MESSAGE, Summary
from tests.fake_mcp import FakeNewsMcp, StubMcpClient, make_doc

pytestmark = pytest.mark.anyio


def _state(*ids: str) -> dict:
    return {
        "query": "ai in healthcare",
        "article_refs": [{"article_id": i, "score": 1.0} for i in ids],
    }


class TestBuildContext:
    async def test_content_capped_at_1000_chars(self) -> None:
        blocks = await Summary._build_context([make_doc("a1", content="x" * 1500)])  # type: ignore[list-item]
        content = blocks[0].split("content: ", 1)[1]
        assert len(content) == 1000

    async def test_article_metadata_in_block(self) -> None:
        (block,) = await Summary._build_context([make_doc("a1")])  # type: ignore[list-item]
        assert block.startswith("Article 0:")
        assert "title: Title a1" in block and "source:TechNews" in block
        assert "published at: 2026-11-01T10:00:00Z" in block

    async def test_missing_content_is_empty(self) -> None:
        doc = make_doc("a1")
        del doc["content"]
        (block,) = await Summary._build_context([doc])  # type: ignore[list-item]
        assert block.endswith("content: ")

    async def test_no_hits_no_blocks(self) -> None:
        assert await Summary._build_context([]) == []


class TestSummaryCall:
    async def test_returns_summary_hits_and_timing(self, make_llm, fake_mcp: FakeNewsMcp) -> None:
        out = await Summary(make_llm(["The summary."]), fake_mcp.client())(_state("a1"))
        assert out["summary"] == "The summary."
        assert [h["article_id"] for h in out["article_hits"]] == ["a1"]
        assert set(out["duration_seconds"]) == {"MGET_CAL", "SUMMARY_LLM_CALL"}
        assert fake_mcp.calls == [("mget", {"ids": ["a1"]})]

    async def test_missing_ids_reply_search_unavailable(self, make_llm, fake_mcp: FakeNewsMcp) -> None:
        out = await Summary(make_llm(["unused"]), fake_mcp.client())(_state("does-not-exist"))
        assert out["summary"] == SEARCH_UNAVAILABLE_MESSAGE and out["article_hits"] == []
        assert "MGET_CAL" in out["duration_seconds"]

    async def test_unexpected_server_crash_reply_has_no_internals(self, make_llm, fake_mcp: FakeNewsMcp) -> None:
        fake_mcp.crash = {"mget"}
        out = await Summary(make_llm(["unused"]), fake_mcp.client())(_state("a1"))
        assert out["summary"] == SEARCH_UNAVAILABLE_MESSAGE
        assert "elasticsearch" not in out["summary"]

    @pytest.mark.parametrize(
        "stub",
        [
            StubMcpClient(exc=ConnectionError("es-mcp-lb:9599 refused")),
            StubMcpClient(
                CallToolResult(content=[TextContent(type="text", text="x")], structured_content=None, meta=None, is_error=True)
            ),
            StubMcpClient(CallToolResult(content=[], structured_content=None, meta=None)),
            StubMcpClient(
                CallToolResult(
                    content=[ImageContent(type="image", data="AAAA", mimeType="image/png")],
                    structured_content=None,
                    meta=None,
                )
            ),
        ],
        ids=["transport_error", "is_error", "no_structured_content", "non_text_block"],
    )
    async def test_mget_failures_reply_search_unavailable(self, make_llm, stub: StubMcpClient) -> None:
        out = await Summary(make_llm(["unused"]), stub)(_state("a1"))  # type: ignore[arg-type]
        assert out == {
            "summary": SEARCH_UNAVAILABLE_MESSAGE,
            "article_hits": [],
            "duration_seconds": {"MGET_CAL": out["duration_seconds"]["MGET_CAL"]},
        }

    async def test_llm_failure_replies_summary_failed(self, fake_mcp: FakeNewsMcp) -> None:
        def _boom(_: object) -> str:
            raise TimeoutError("openai timed out")

        out = await Summary(RunnableLambda(_boom), fake_mcp.client())(_state("a1"))  # type: ignore[arg-type]
        assert out["summary"] == SUMMARY_FAILED_MESSAGE and out["article_hits"] == []
        assert "SUMMARY_LLM_CALL" in out["duration_seconds"]
