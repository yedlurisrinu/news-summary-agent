"""``GET /api/mcp/check`` against a real, auth-protected MCP server (mcp 2.3.0) on loopback.

The server mirrors es-mcp-server's shape: a ``search`` tool, ``validate_token_resource``,
and a verifier that only accepts the access token the fake Keycloak issued.
"""

from __future__ import annotations

import socket
import threading
import time
import warnings
from collections.abc import Iterator
from typing import Any

import pytest
import uvicorn
from fastapi.testclient import TestClient

from tests.auth.fake_keycloak import FakeKeycloak
from tests.auth.login_flow_test import MemoryStore, _build, _login

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from mcp.server.auth.provider import AccessToken
    from mcp.server.auth.settings import AuthSettings
    from mcp.server.mcpserver import MCPServer


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


@pytest.fixture()
def mcp_server() -> Iterator[tuple[str, list[str]]]:
    port = _free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    seen: list[str] = []

    class Verifier:
        async def verify_token(self, token: str) -> AccessToken | None:
            seen.append(token)
            if token != "access-1":
                return None
            return AccessToken(token=token, client_id="cimd", scopes=[], resource=url)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        server = MCPServer(
            "es-mcp-server-test",
            token_verifier=Verifier(),
            auth=AuthSettings(
                issuer_url="http://keycloak:8180/realms/es-mcp",
                resource_server_url=url,
                validate_token_resource=True,
            ),
        )

    @server.tool()
    def search(query: str) -> str:
        """dummy search"""
        return "success"

    app = server.streamable_http_app(stateless_http=True)
    uv = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if uv.started:
            break
        time.sleep(0.05)
    yield url, seen
    uv.should_exit = True
    thread.join(timeout=5)


def _client_for(mcp_url: str, kc: FakeKeycloak, store: MemoryStore) -> TestClient:
    app = _build(kc, store)
    cfg: Any = app.state.bff_config
    object.__setattr__(cfg, "mcp_url", mcp_url)  # point the app at the test MCP server
    return TestClient(app, base_url="http://news-summary-app:8007")


def test_mcp_check_with_user_token(mcp_server: tuple[str, list[str]]) -> None:
    url, seen = mcp_server
    kc, store = FakeKeycloak(), MemoryStore()
    client = _client_for(url, kc, store)
    _login(client, kc)
    r = client.get("/api/mcp/check")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tools"] == ["search"]
    assert body["user"] == "user-123"
    assert "success" in str(body["call"]["search"])
    assert seen and set(seen) == {
        "access-1"
    }  # the user's own token reached the MCP server


def test_mcp_check_requires_login(mcp_server: tuple[str, list[str]]) -> None:
    url, seen = mcp_server
    client = _client_for(url, FakeKeycloak(), MemoryStore())
    assert client.get("/api/mcp/check").status_code == 401
    assert seen == []  # nothing reached the MCP server


def test_mcp_rejects_token_gives_502_and_keeps_session(
    mcp_server: tuple[str, list[str]],
) -> None:
    url, _ = mcp_server
    kc, store = FakeKeycloak(), MemoryStore()
    client = _client_for(url, kc, store)
    _login(client, kc)
    sessions: Any = client.app.state.sessions  # type: ignore[attr-defined]
    sid = next(iter(sessions._live))
    sessions._live[sid].access_token = "not-the-users-token"
    assert client.get("/api/mcp/check").status_code == 502
    assert sid in store.data  # an MCP failure is not an auth failure
