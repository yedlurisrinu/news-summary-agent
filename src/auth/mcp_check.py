"""``GET /api/mcp/check`` - Gate 1 proof: the logged-in user's token reaches es-mcp-server.

Lists the MCP tools through ``langchain.mcp.MCPAdapter`` (fastmcp underneath) and calls the
dummy ``search`` tool. The graph is untouched until Gate 2 (S890).

The refresh happens BEFORE the call (``SessionStore.access_token``); the bearer auth below
only attaches the header. A 401 from es-mcp-server would surface as a generic MCPError
(L70), which is why refreshing up front is simpler than retrying on 401.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx2
from fastapi import APIRouter, Request
from fastmcp import Client
from starlette.responses import JSONResponse

from langchain_core._api import suppress_langchain_beta_warning

from auth.sessions import AuthUnavailable, LoginRequired

with suppress_langchain_beta_warning():  # langchain.mcp warns once that it is in beta
    from langchain.mcp import MCPAdapter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["MCP"])

CHECK_TOOL = "search"
CHECK_ARGS: dict[str, Any] = {"query": "mcp check"}


class BearerAuth(httpx2.Auth):
    """Attach one already-fresh access token to every MCP HTTP request."""

    def __init__(self, access_token: str) -> None:
        self._header = f"Bearer {access_token}"

    async def async_auth_flow(
        self, request: httpx2.Request
    ) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        request.headers["Authorization"] = self._header
        yield request


@router.get("/api/mcp/check")
async def mcp_check(request: Request) -> JSONResponse:
    sessions = request.app.state.sessions
    mcp_url: str = request.app.state.bff_config.mcp_url
    sid: str = request.session["sid"]  # the guard guarantees a live session here
    try:
        access_token = await sessions.access_token(sid)
    except LoginRequired:
        return JSONResponse({"error": "login_required"}, status_code=401)
    except AuthUnavailable:
        return JSONResponse({"error":"auth_unavailable"}, status_code=503, headers={"Retry-After":"30"})

    try:
        client = Client(mcp_url, auth=BearerAuth(access_token))
        async with MCPAdapter(client) as adapter:
            tools = await adapter.list_tools()
            names = [tool.name for tool in tools]
            result: Any = None
            for tool in tools:
                if tool.name == CHECK_TOOL:
                    result = await tool.ainvoke(CHECK_ARGS)
                    break
    except Exception as exc:  # MCP failure is not an auth failure: keep the session
        logger.warning("mcp check failed: %s: %s", type(exc).__name__, exc)
        return JSONResponse({"error": "mcp_unavailable"}, status_code=502)

    return JSONResponse(
        {"user": sessions.subject(sid), "mcp_url": mcp_url, "tools": names, "call": {CHECK_TOOL: result}}
    )
