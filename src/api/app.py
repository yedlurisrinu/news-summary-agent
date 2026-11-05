"""FastAPI application (port 8007): the news-summary workflow behind a BFF login.

Endpoints:
  POST /query    — run the classify → search → summarize workflow for one query
  GET  /chat     — chat UI (static/chat-ui.html)
  GET  /login    — login page (static/login.html)
  GET  /health   — liveness probe (public)
  /auth/login, /auth/callback — Keycloak login (see auth/routes.py)

Every path except /login, /auth/login, /auth/callback and /health requires a session
(``DEFAULT_PUBLIC_PATHS`` in auth/guard.py).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import Depends, FastAPI, Request, Response
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from adapters.vault_session_store import VaultRefreshTokenStore
from auth.auth_server_metadata import AuthServerMetadata
from auth.config import BffConfig
from auth.discovery import discover_issuer
from auth.guard import RETRY_AFTER_SECONDS
from auth.wiring import install_auth
from config.application_config import get_config, load_config
from config.logging_setup import setup_logging
from config.vault_secrets import load_secrets
from data.news_input import NewsInput
from data.query_failed_response import QueryFailedResponse
from data.query_request import QueryRequest
from data.query_success_response import QuerySuccessResponse
from graph.run_graph_workflow import run_graph

setup_logging()
logger = logging.getLogger(__name__)

load_secrets()
load_config()

from adapters.lifespan import lifespan

app = FastAPI(
    title="News Summary Workflow",
    description="News summary workflow: classify the query → search articles (es-mcp-server) → summarize with citations",
    version="1.0.0",
    lifespan=lifespan,
)


def get_graph(request: Request) -> CompiledStateGraph[Any, None, Any, Any]:
    """FastAPI dependency: return the compiled graph from application state."""
    return cast("CompiledStateGraph[Any, None, Any, Any]", request.app.state.graph)


as_meta = AuthServerMetadata(discover_issuer)
app.state.as_meta = as_meta

# BFF login: default-deny guard + Keycloak login + server-side sessions (tokens never reach the browser).
# CORS removed: the chat page and the API share one origin (http://news-summary-app:8007).

session_store = VaultRefreshTokenStore(
    os.environ["VAULT_ADDR"],
    os.environ["VAULT_TOKEN"],
    get_config("APP_VAULT_SESSIONS_PATH"),
)
app.state.session_store = session_store  # the lifespan sweeps expired entries from it
install_auth(app, BffConfig.load(), session_store, as_meta)

# static/ sits at the repo root, next to src/
STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/login")
def login() -> FileResponse:
    "Serves the login page"
    return FileResponse(STATIC_DIR / "login.html")


@app.get("/chat")
def serve_ui() -> FileResponse:
    """Serve the static chat UI (contextless HTML)."""
    return FileResponse(STATIC_DIR / "chat-ui.html")


@app.get("/health", tags=["Ops"])
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post(
    "/query",
    response_model=Annotated[
        QuerySuccessResponse | QueryFailedResponse,
        Field(discriminator="status"),
    ],
    responses={
        500: {
            "model": QueryFailedResponse,
            "description": "Unexpected failure (fixed text)",
        },
        503: {
            "model": QueryFailedResponse,
            "description": "Search (es-mcp-server) unavailable; retry after Retry-After seconds",
        },
    },
    tags=["Pipeline"],
)
async def query(
    request: QueryRequest,
    graph: Annotated[CompiledStateGraph[Any, None, Any, Any], Depends(get_graph)],
    http_response: Response,
) -> QuerySuccessResponse | QueryFailedResponse:
    """
    Run the workflow for one query: articles come from es-mcp-server's MCP tools, the summary from the LLM.

    200 with ``status: ok`` on success; on failure the body is a ``QueryFailedResponse`` and the
    HTTP status is its ``error_code`` (503 + Retry-After when search is unavailable, else 500).
    """
    logger.info("POST /query  query=%r", request.query)
    user_input = NewsInput(query=request.query)
    response: QuerySuccessResponse | QueryFailedResponse = await run_graph(
        user_input, graph
    )
    if isinstance(response, QueryFailedResponse):
        http_response.status_code = response.error_code
        if response.error_code == 503:
            http_response.headers["Retry-After"] = RETRY_AFTER_SECONDS
    return response
