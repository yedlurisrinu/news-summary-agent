import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from adapters.llm import get_chat_model
from adapters.mcp_session import LazyMcpSession
from adapters.postgres import postgres_pool
from auth.service_account_token import ServiceAccountTokenProvider
from auth.service_token_auth import ServiceTokenAuth
from config.application_config import get_config
from graph.build_graph import build

logger = logging.getLogger(__name__)


async def _sweep_sessions(store: Any, max_age_sec: int, interval_sec: int) -> None:
    """Background task: remove expired session entries at start-up, then every interval."""
    while True:
        try:
            removed = await store.sweep(max_age_sec)
            logger.info("session store sweep: %d expired entries removed", removed)
        except Exception:
            logger.exception("session store sweep failed; retrying next interval")
        await asyncio.sleep(interval_sec)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[Any, Any]:
    """
    Open the Postgres pool (checkpointer) and build the chat model and the MCP client,
    then put the compiled graph on ``app.state.graph``.

    The MCP client is created here but NOT connected: ``LazyMcpSession`` connects on the
    first tool call and reuses that connection (see adapters/mcp_session.py). Startup
    therefore does not depend on es-mcp-server or Keycloak being up.
    """
    as_meta = app.state.as_meta
    provider = ServiceAccountTokenProvider(
        get_config("SERVICE_CLIENT_ID"),
        os.environ["SERVICE_CLIENT_SECRET"],
        token_endpoint=as_meta.token_endpoint,
    )
    mcp_client = LazyMcpSession(
        Client(
            StreamableHttpTransport(get_config("MCP_URL"), auth=ServiceTokenAuth(provider)),
            # pinned: skips the per-connection server/discover version probe
            mode=get_config("MCP_PROTOCOL_VERSION"),
        )
    )
    llm = get_chat_model(
        get_config("LLM_MODEL"),
        float(get_config("LLM_TEMPERATURE")),
        int(get_config("LLM_TIMEOUT_SEC")),
    )
    async with postgres_pool() as pool:
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        app.state.graph = await build(saver, llm, mcp_client)
        store = getattr(app.state, "session_store", None)
        sweeper = (
            asyncio.create_task(
                _sweep_sessions(
                    store,
                    int(get_config("SESSION_STORE_MAX_AGE_SEC")),
                    int(get_config("SESSION_SWEEP_INTERVAL_SEC")),
                )
            )
            if store is not None
            else None
        )
        try:
            yield
        finally:
            if sweeper is not None:
                sweeper.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sweeper
            await mcp_client.aclose()
