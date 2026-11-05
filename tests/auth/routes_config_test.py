"""
tests/auth/routes_config_test.py

/auth/callback error mapping (only fixed codes reach the URL) and BffConfig.load:
  - Keycloak access_denied, other OAuth errors and unexpected failures → /login?error=...
  - BffConfig fails fast on a short session secret and defaults the session max age
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from authlib.integrations.base_client import OAuthError
from fastapi.testclient import TestClient

from auth.config import SESSION_SECRET_ENV, BffConfig
from tests.auth.fake_keycloak import ISSUER, FakeKeycloak
from tests.auth.login_flow_test import _login


def _callback_with(client: TestClient, kc: FakeKeycloak, exc: Exception) -> str:
    _login(client, kc)  # discovery + client registration, so keycloak.issuer is known
    registered = client.app.state.keycloak._client  # type: ignore[attr-defined]
    with patch.object(registered, "authorize_access_token", AsyncMock(side_effect=exc)):
        r = client.get(
            "/auth/callback",
            params={"code": "c", "state": "s", "iss": ISSUER},
            follow_redirects=False,
        )
    assert r.status_code == 302
    return str(r.headers["location"])


class TestCallbackErrors:
    def test_keycloak_access_denied(self, client: TestClient, kc: FakeKeycloak) -> None:
        assert (
            _callback_with(client, kc, OAuthError(error="access_denied"))
            == "/login?error=access_denied"
        )

    def test_other_oauth_error_is_login_failed(
        self, client: TestClient, kc: FakeKeycloak
    ) -> None:
        assert (
            _callback_with(client, kc, OAuthError(error="invalid_grant"))
            == "/login?error=login_failed"
        )

    def test_unexpected_error_is_login_failed(
        self, client: TestClient, kc: FakeKeycloak
    ) -> None:
        location = _callback_with(client, kc, RuntimeError("keycloak:8180 exploded"))
        assert location == "/login?error=login_failed"


class TestBffConfigLoad:
    def test_short_secret_fails_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SESSION_SECRET_ENV, "too-short")
        with pytest.raises(ValueError, match=SESSION_SECRET_ENV):
            BffConfig.load()

    def test_loads_from_config_and_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SESSION_SECRET_ENV, "s" * 48)
        cfg = BffConfig.load()
        assert cfg.base_url == "http://news-summary-app:8007"
        assert cfg.redirect_uri == "http://news-summary-app:8007/auth/callback"
        assert (
            cfg.mcp_url == "http://es-mcp-lb:9599/mcp"
            and cfg.session_max_age_sec == 36000
        )

    def test_missing_max_age_uses_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import config.application_config as app_config

        monkeypatch.setenv(SESSION_SECRET_ENV, "s" * 48)
        monkeypatch.delitem(app_config.app_config, "SESSION_MAX_AGE_SEC")
        assert BffConfig.load().session_max_age_sec == 36000


class TestCallbackSessionStore:
    def _callback(self, client: TestClient) -> tuple[int, str, bool]:
        r = client.get(
            "/auth/callback",
            params={"code": "c", "state": "s", "iss": ISSUER},
            follow_redirects=False,
        )
        return (
            r.status_code,
            str(r.headers.get("location")),
            "nsa_session" in r.headers.get("set-cookie", ""),
        )

    def test_session_store_down_redirects_to_auth_unavailable(
        self, client: TestClient, kc: FakeKeycloak, store
    ) -> None:
        _login(client, kc)  # registers the client and learns the issuer
        client.cookies.clear()
        token = {
            "access_token": "at",
            "expires_in": 300,
            "refresh_token": "rt",
            "userinfo": {"sub": "u1"},
        }
        registered = client.app.state.keycloak._client  # type: ignore[attr-defined]
        with (
            patch.object(
                registered, "authorize_access_token", AsyncMock(return_value=token)
            ),
            patch.object(
                store,
                "save",
                AsyncMock(side_effect=ConnectionError("vault-server:8200 refused")),
            ),
        ):
            status, location, _ = self._callback(client)
        assert (status, location) == (302, "/login?error=auth_unavailable")
        assert (
            client.get(
                "/chat", headers={"accept": "text/html"}, follow_redirects=False
            ).headers["location"]
            == "/login"
        )

    def test_token_without_refresh_token_redirects_to_login_failed(
        self, client: TestClient, kc: FakeKeycloak
    ) -> None:
        _login(client, kc)
        client.cookies.clear()
        registered = client.app.state.keycloak._client  # type: ignore[attr-defined]
        with patch.object(
            registered,
            "authorize_access_token",
            AsyncMock(return_value={"access_token": "at", "expires_in": 300}),
        ):
            status, location, _ = self._callback(client)
        assert (status, location) == (302, "/login?error=login_failed")
