"""
tests/api/app_test.py

Tests for api/app.py via Starlette TestClient.

api/app.py runs setup_logging(), load_secrets(), load_config() and
VaultRefreshTokenStore() at import time; this module patches the calls that need
external services (Vault, Keycloak) before the first import, and
overrides the get_graph dependency with a fake compiled graph so no
Postgres / ES / LLM is needed. The lifespan is never entered (TestClient
is used without its context manager), so no live connections are opened.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

STATE = {
    "query": "ai news",
    "categories": ["ai"],
    "summary": "A grounded summary.",
    "article_hits": [
        {
            "title": "AI Healthcare",
            "source": "TechNews",
            "published_at": "2026-03-28",
            "link": "https://example.com/ai",
        }
    ],
    "duration_seconds": {"CLASSIFY": 0.1, "SEARCH": 0.2},
}


@pytest.fixture()
def app_module():
    """Import api.app fresh with the external-service startup calls patched out."""
    # The BFF login is tested in tests/auth; here it is switched off so these tests keep
    # exercising the graph endpoints directly (no Vault secret, no Keycloak).
    with (
        patch("config.vault_secrets.load_secrets"),
        patch("auth.wiring.install_auth"),
        patch("auth.config.BffConfig.load"),
        patch("adapters.vault_session_store.VaultRefreshTokenStore"),
        patch.dict(os.environ, {"VAULT_ADDR": "x", "VAULT_TOKEN": "x"}),
    ):
        sys.modules.pop("api.app", None)
        import api.app as m

    yield m
    m.app.dependency_overrides.clear()
    sys.modules.pop("api.app", None)


def _client(app_module):
    from fastapi.testclient import TestClient

    return TestClient(app_module.app)


class TestHealth:
    def test_health_ok(self, app_module):
        resp = _client(app_module).get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestServeUi:
    def test_returns_file_response(self, app_module):
        from starlette.responses import FileResponse

        resp = app_module.serve_ui()
        assert isinstance(resp, FileResponse)
        assert str(resp.path).endswith("static/chat-ui.html")

    def test_login_page_is_file_response(self, app_module):
        from starlette.responses import FileResponse

        resp = app_module.login()
        assert isinstance(resp, FileResponse)
        assert str(resp.path).endswith("static/login.html")

    def test_static_dir_exists(self, app_module):
        assert (app_module.STATIC_DIR / "chat-ui.html").is_file()


class TestGetGraph:
    def test_returns_graph_from_app_state(self, app_module):
        request = MagicMock()
        request.app.state.graph = sentinel = object()
        assert app_module.get_graph(request) is sentinel


class TestQueryEndpoint:
    def test_query_success(self, app_module):
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value=STATE)
        app_module.app.dependency_overrides[app_module.get_graph] = lambda: graph

        resp = _client(app_module).post("/query", json={"query": "ai news"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["summary"] == "A grounded summary."

    def test_unexpected_failure_is_http_500_with_fixed_text(self, app_module):
        graph = MagicMock()
        graph.ainvoke = AsyncMock(side_effect=RuntimeError("es-mcp-lb:9599 boom"))
        app_module.app.dependency_overrides[app_module.get_graph] = lambda: graph

        resp = _client(app_module).post("/query", json={"query": "ai news"})

        assert resp.status_code == 500
        body = resp.json()
        assert body["status"] == "failed"
        assert body["error_code"] == 500
        assert "boom" not in body["error"] and "9599" not in body["error"]
        assert "retry-after" not in resp.headers

    def test_search_unavailable_is_http_503_with_retry_after(self, app_module):
        from exception.search_unavailable import (
            SEARCH_UNAVAILABLE_MESSAGE,
            SearchUnavailable,
        )

        graph = MagicMock()
        graph.ainvoke = AsyncMock(side_effect=SearchUnavailable())
        app_module.app.dependency_overrides[app_module.get_graph] = lambda: graph

        resp = _client(app_module).post("/query", json={"query": "ai news"})

        assert resp.status_code == 503 and resp.headers["retry-after"] == "30"
        assert resp.json() == {
            "query": "ai news",
            "error": SEARCH_UNAVAILABLE_MESSAGE,
            "error_code": 503,
            "status": "failed",
        }

    def test_openapi_declares_500_and_503(self, app_module):
        responses = app_module.app.openapi()["paths"]["/query"]["post"]["responses"]
        assert {"200", "500", "503"} <= set(responses) and "502" not in responses
