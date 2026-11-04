from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI
from adapters.postgres import connect_postgres
from graph.build_graph import build
from adapters.llm import get_chat_model
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from config.application_config import get_config
from auth.service_account_token import ServiceAccountTokenProvider
from fastmcp import Client
from auth.service_token_auth import ServiceTokenAuth
from fastmcp.client.transports import StreamableHttpTransport
import os


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[Any, Any]:
    """
    A context manager to maintain postgres saver connection , model connection, elasticsearch connection
    and inject into graph.
    """
    as_meta = app.state.as_meta
    provider = ServiceAccountTokenProvider(
        get_config("SERVICE_CLIENT_ID"),
        os.environ["SERVICE_CLIENT_SECRET"],
        token_endpoint=as_meta.token_endpoint,
    )
    mcp_client = Client(
        StreamableHttpTransport(get_config("MCP_URL"), auth=ServiceTokenAuth(provider))
    )
    async with await connect_postgres() as pool:
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        app.state.graph = await build(saver, await get_chat_model(), mcp_client)
        yield
