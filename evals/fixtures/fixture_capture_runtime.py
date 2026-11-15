"""Clients for an e2e capture run: the real LLM, the eval MCP server and the frozen ES index."""

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from httpx2 import URL, AsyncClient
from langchain_core.language_models import BaseChatModel

from adapters.llm import get_chat_model
from adapters.mcp_session import LazyMcpSession
from auth.auth_server_metadata import AuthServerMetadata
from auth.discovery import discover_issuer
from auth.service_account_token import ServiceAccountTokenProvider
from auth.service_token_auth import ServiceTokenAuth
from evals.eval_config.load_eval_config import get_config

log = logging.getLogger(__name__)


def es_client() -> AsyncClient:
    """HTTP client for the eval Elasticsearch (API key from ES_NEWS_SUMMARY_EVAL_API_KEY)."""
    return AsyncClient(
        base_url=URL(get_config("EVAL_ES_URL")),
        headers={
            "Authorization": f"ApiKey {os.environ['ES_NEWS_SUMMARY_EVAL_API_KEY']}"
        },
    )


@dataclass
class EvalRuntime:
    """Everything a capture run needs: its id, the frozen index metadata and the open clients."""

    run_id: str
    meta: dict[str, Any]
    llm: BaseChatModel
    mcp_client: LazyMcpSession
    es_client: AsyncClient
    frozen_index: str
    offset_ms: int
    mcp_url: str


@asynccontextmanager
async def open_runtime(run_id: str) -> AsyncIterator[EvalRuntime]:
    """Preflight + per-run clients; both clients are closed when the run ends, even on error."""
    es_http_client = es_client()
    rt: EvalRuntime | None = None
    try:
        rt = await _build_runtime(run_id=run_id, es_http_client=es_http_client)
        yield rt
    finally:
        if rt is not None:
            await rt.mcp_client.aclose()
        await es_http_client.aclose()


async def _build_runtime(run_id: str, es_http_client: AsyncClient) -> EvalRuntime:
    """Preflight the frozen index, shift its dates to now, and build the LLM and MCP clients.

    Raises ValueError when the index _meta / runtime mapping lacks reference_now,
    reference_now_ms or published_at; HTTP errors from Elasticsearch propagate.
    """
    frozen_eval_index = str(get_config("ES_FROZEN_EVAL_INDEX"))
    eval_mcp_url = str(get_config("EVAL_MCP_URL"))
    mapping_meta = await es_http_client.get(f"{frozen_eval_index}/_mapping/")
    mapping_meta.raise_for_status()
    response_payload: dict[str, Any] = mapping_meta.json()
    mapping = response_payload[frozen_eval_index]["mappings"]
    meta: dict[str, Any] = mapping["_meta"]

    # frozen index preflight: all three are needed to shift published_at to "now"
    reference_now = meta.get("reference_now")
    reference_now_ms = meta.get("reference_now_ms")
    published_at = mapping.get("runtime", {}).get("published_at")
    if not (reference_now and reference_now_ms and published_at):
        raise ValueError(
            "frozen index _meta / runtime is missing reference_now, reference_now_ms or published_at"
        )
    runtime = mapping["runtime"]
    log.debug("frozen index runtime mapping: %s", runtime)
    offset_ms = time.time_ns() // 1_000_000 - int(reference_now_ms)
    runtime["published_at"]["script"]["params"]["offset_ms"] = offset_ms

    llm = get_chat_model(
        get_config("LLM_MODEL"),
        float(get_config("LLM_TEMPERATURE")),
        int(get_config("LLM_TIMEOUT_SEC")),
    )
    put_response = await es_http_client.put(
        f"{frozen_eval_index}/_mapping", json={"runtime": runtime}
    )
    put_response.raise_for_status()
    log.info(
        "runtime mapping updated on %s: %s", frozen_eval_index, put_response.json()
    )

    provider = ServiceAccountTokenProvider(
        get_config("SERVICE_CLIENT_ID"),
        os.environ["SERVICE_CLIENT_SECRET"],
        token_endpoint=AuthServerMetadata(discover_issuer).token_endpoint,
    )
    mcp_client = LazyMcpSession(
        Client(
            StreamableHttpTransport(eval_mcp_url, auth=ServiceTokenAuth(provider)),
            # pinned: skips the per-connection server/discover version probe
            mode=get_config("MCP_PROTOCOL_VERSION"),
        )
    )

    return EvalRuntime(
        run_id=run_id,
        meta=meta,
        llm=llm,
        mcp_client=mcp_client,
        es_client=es_http_client,
        frozen_index=frozen_eval_index,
        offset_ms=offset_ms,
        mcp_url=eval_mcp_url,
    )
