"""
@Author: Srini Yedluri
@Date: 3/27/26
@Time: 11:57 AM
@File: app.py
"""

from __future__ import annotations

from data.news_input import NewsInput

"""
api/app.py

FastAPI application — listens on port 8007.

Endpoints:
  POST /query    — run the Graph with multi-node
  GET  /health   — liveness probe
"""
from pydantic import Field
from pathlib import Path
from typing import Annotated, Any, Union, cast
from graph.run_graph_workflow import run_graph
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles
from data.query_success_response import QuerySuccessResponse
from data.query_failed_response import QueryFailedResponse
from config.logging_setup import setup_logging
from config.vault_secrets import load_secrets
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from data.query_request import QueryRequest
from langgraph.graph.state import CompiledStateGraph
import logging
from config.application_config import load_config
from adapters.elasticsearch import get_es_client

setup_logging()
logger = logging.getLogger(__name__)

load_secrets()
load_config()
get_es_client()
from adapters.lifespan import lifespan

app = FastAPI(
    title="News Summary Workflow",
    description="Agent Graph pipeline: Category Search → Semantic Search → Summarise -> Email (TODO)",
    version="1.0.0",
    lifespan=lifespan
)
def get_graph(request: Request) -> CompiledStateGraph[Any, None, Any, Any]:
    """FastAPI dependency: return the compiled graph from application state."""
    return cast("CompiledStateGraph[Any, None, Any, Any]", request.app.state.graph)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
chat_ui_path = Path(__name__).parent.parent

""" End point that will render chat ui html for chat interaction, this is context less """


@app.get("/login")
def login() -> FileResponse:
    "Serves the login page"
    return FileResponse(str(chat_ui_path)+"/static/login.html")


@app.get("/")
def serve_ui() -> FileResponse:
    """Serve the static chat UI (contextless HTML)."""
    return FileResponse(str(chat_ui_path) + "/static/chat-ui.html")

@app.get("/health", tags=["Ops"])
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}

@app.post(
    "/query",
    response_model=Annotated[
        Union[QuerySuccessResponse, QueryFailedResponse],
        Field(discriminator="status"),
    ],
    responses={502: {"model": QueryFailedResponse, "description": "Agent failed"}},
    tags=["Pipeline"],
)
def query(request: QueryRequest, graph: Annotated[CompiledStateGraph[Any, None, Any, Any], Depends(get_graph)]) -> QuerySuccessResponse | QueryFailedResponse:
    """
    Run the full agent graph news summary for a query using ElasticSearch and Qdrant as RAG backend.
    """
    logger.info("POST /query  query=%r", request.query)
    user_input = NewsInput(query=request.query)
    response: QuerySuccessResponse | QueryFailedResponse = run_graph(
    user_input, graph)
    return response

