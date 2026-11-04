"""
tests/api/app_test.py

Tests for api/app.py via Starlette TestClient.

api/app.py runs setup_logging(), load_secrets(), load_config() and
get_es_client() at import time; this module patches the two that need
external services (Vault, Elasticsearch) before the first import, and
overrides the get_graph dependency with a fake compiled graph so no
Postgres / ES / LLM is needed. The lifespan is never entered (TestClient
is used without its context manager), so no live connections are opened.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

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
        patch("adapters.elasticsearch.get_es_client"),
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
        assert resp.path.endswith("chat-ui.html")


class TestQueryEndpoint:
    def test_query_success(self, app_module):
        graph = MagicMock()
        graph.invoke.return_value = STATE
        app_module.app.dependency_overrides[app_module.get_graph] = lambda: graph

        resp = _client(app_module).post("/query", json={"query": "ai news"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["summary"] == "A grounded summary."

    def test_query_failure_maps_to_failed_body(self, app_module):
        graph = MagicMock()
        graph.invoke.side_effect = RuntimeError("boom")
        app_module.app.dependency_overrides[app_module.get_graph] = lambda: graph

        resp = _client(app_module).post("/query", json={"query": "ai news"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failed"
        assert body["error_code"] == 500
