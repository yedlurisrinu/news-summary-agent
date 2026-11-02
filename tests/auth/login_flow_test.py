"""End-to-end tests of the BFF login against a fake Keycloak (no network, no Vault).

A minimal FastAPI app gets ``install_auth`` exactly as ``api/app.py`` does, with an
in-memory refresh-token store and a discovery stub. Covers guide step 11, U1-U12.
"""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi.responses import JSONResponse
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.config import BffConfig
from auth.ports import StoredSession
from auth.sessions import LoginRequired, SessionStore
from auth.wiring import install_auth
from tests.auth.fake_keycloak import CLIENT_ID, ISSUER, MCP_URL, FakeKeycloak

BASE = "http://news-summary-app:8007"


class MemoryStore:
    """In-memory RefreshTokenStore (stands in for the Vault implementation)."""

    def __init__(self) -> None:
        self.data: dict[str, StoredSession] = {}

    async def get(self, sid: str) -> StoredSession | None:
        return self.data.get(sid)

    async def save(self, sid: str, session: StoredSession) -> None:
        self.data[sid] = session

    async def delete(self, sid: str) -> None:
        self.data.pop(sid, None)


def _config() -> BffConfig:
    return BffConfig(
        base_url=BASE, mcp_url=MCP_URL, client_id=CLIENT_ID, allowed_issuer=ISSUER,
        session_secret="s" * 48,
    )


def _build(kc: FakeKeycloak, store: MemoryStore, issuer: str = ISSUER) -> FastAPI:
    app = FastAPI()

    @app.get("/chat")
    def home() -> dict[str, str]:
        return {"page": "chat"}

    @app.post("/query")
    def query() -> dict[str, str]:
        return {"answer": "ok"}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}
    
    @app.get("/cached")
    def cached() -> JSONResponse:  # a route that tries to allow caching
        return JSONResponse({"page": "cached"}, headers={"Cache-Control": "max-age=600"})

    @app.get("/login")
    def login_page() -> dict[str, str]:
        return {"page": "login"}

    async def discover() -> str:
        return issuer

    install_auth(app, _config(), store, discover, http_transport=kc.transport)
    return app


def _login(client: TestClient, kc: FakeKeycloak) -> dict[str, str]:
    """Drive /auth/login -> (Keycloak) -> /auth/callback; return the authorize params."""
    r = client.get("/auth/login", follow_redirects=False)
    assert r.status_code == 302
    params = {k: v[0] for k, v in parse_qs(urlparse(r.headers["location"]).query).items()}
    kc.nonce = params["nonce"]
    r = client.get(
        "/auth/callback",
        params={"code": "abc", "state": params["state"], "iss": ISSUER},
        follow_redirects=False,
    )
    assert r.status_code == 302 and r.headers["location"] == "/chat", r.headers.get("location")
    return params


@pytest.fixture()
def kc() -> FakeKeycloak:
    return FakeKeycloak()


@pytest.fixture()
def store() -> MemoryStore:
    return MemoryStore()


@pytest.fixture()
def client(kc: FakeKeycloak, store: MemoryStore) -> TestClient:
    return TestClient(_build(kc, store), base_url=BASE)


class TestGuard:
    def test_public_paths_open(self, client: TestClient) -> None:  # U1
        assert client.get("/health").status_code == 200
        assert client.get("/login").status_code == 200

    def test_page_without_session_redirects_to_login(self, client: TestClient) -> None:  # U2
        r = client.get("/chat", headers={"accept": "text/html"}, follow_redirects=False)
        assert r.status_code == 302 and r.headers["location"] == "/login"

    def test_api_without_session_is_401(self, client: TestClient) -> None:  # U2
        r = client.post("/query", json={})
        assert r.status_code == 401 and r.json() == {"error": "login_required"}

    def test_unknown_sid_is_401(self, client: TestClient, store: MemoryStore) -> None:
        # a forged sid cannot be planted: the cookie is signed, so tamper -> empty session
        client.cookies.set("nsa_session", "garbage")
        assert client.post("/query", json={}).status_code == 401

class TestNoStore:  # F-E2E6-2
    def test_refused_page_and_api_are_no_store(self, client: TestClient) -> None:  # U13
        r = client.get("/", headers={"accept": "text/html"}, follow_redirects=False)
        assert r.status_code == 302 and r.headers["cache-control"] == "no-store"
        r = client.post("/query", json={})
        assert r.status_code == 401 and r.headers["cache-control"] == "no-store"

    def test_allowed_page_is_no_store_and_overrides_route(self, client: TestClient, kc: FakeKeycloak) -> None:  # U14
        _login(client, kc)
        r = client.get("/chat", headers={"accept": "text/html"})
        assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
        r = client.get("/cached")
        assert r.status_code == 200 and r.headers.get_list("cache-control") == ["no-store"]

    def test_public_paths_untouched(self, client: TestClient) -> None:  # U15
        assert "cache-control" not in client.get("/login").headers
        assert "cache-control" not in client.get("/health").headers

class TestLogin:
    def test_authorize_redirect_params(self, client: TestClient) -> None:  # U3
        r = client.get("/auth/login", follow_redirects=False)
        url = urlparse(r.headers["location"])
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        assert f"{url.scheme}://{url.netloc}{url.path}" == f"{ISSUER}/protocol/openid-connect/auth"
        assert q["client_id"] == CLIENT_ID
        assert q["redirect_uri"] == f"{BASE}/auth/callback"
        assert q["resource"] == MCP_URL
        assert q["code_challenge_method"] == "S256"
        assert {"state", "nonce", "code_challenge"} <= q.keys()

    def test_wrong_discovered_issuer_never_redirects_to_it(self, kc: FakeKeycloak, store: MemoryStore) -> None:  # U4
        c = TestClient(_build(kc, store, issuer="http://evil:8180/realms/x"), base_url=BASE)
        r = c.get("/auth/login", follow_redirects=False)
        assert r.headers["location"] == "/login?error=login_failed"

    def test_full_login_then_api_works(self, client: TestClient, kc: FakeKeycloak, store: MemoryStore) -> None:
        _login(client, kc)
        assert client.post("/query", json={}).json() == {"answer": "ok"}
        token_call = next(r for r in kc.requests if r["body"].get("grant_type") == "authorization_code")
        assert token_call["body"]["resource"] == MCP_URL
        assert "code_verifier" in token_call["body"]
        [stored] = store.data.values()
        assert stored.sub == "user-123" and stored.refresh_token == "refresh-1"

    def test_callback_wrong_iss(self, client: TestClient) -> None:  # U5
        client.get("/auth/login", follow_redirects=False)
        r = client.get("/auth/callback", params={"code": "c", "state": "s", "iss": "http://evil"}, follow_redirects=False)
        assert r.headers["location"] == "/login?error=login_failed"

    def test_callback_unknown_state(self, client: TestClient) -> None:  # U6
        client.get("/auth/login", follow_redirects=False)
        r = client.get("/auth/callback", params={"code": "c", "state": "nope", "iss": ISSUER}, follow_redirects=False)
        assert r.headers["location"] == "/login?error=state_mismatch"

    def test_callback_access_denied(self, client: TestClient) -> None:  # U7
        r = client.get("/auth/callback", params={"error": "access_denied"}, follow_redirects=False)
        assert r.headers["location"] == "/login?error=access_denied"


def _sessions(client: TestClient) -> SessionStore:
    app: Any = client.app
    sessions: SessionStore = app.state.sessions
    return sessions


class TestSessions:
    def test_valid_token_not_refreshed(self, client: TestClient, kc: FakeKeycloak) -> None:  # U8
        _login(client, kc)
        sid = next(iter(_sessions(client)._live))
        before = len(kc.requests)
        assert asyncio.run(_sessions(client).access_token(sid)) == "access-1"
        assert len(kc.requests) == before

    def test_near_expiry_refreshes_with_resource_and_rotation(self, client: TestClient, kc: FakeKeycloak, store: MemoryStore) -> None:  # U9
        kc.expires_in = 10  # inside the 30 s skew -> refresh on next use
        kc.rotate_refresh = True
        _login(client, kc)
        sid = next(iter(_sessions(client)._live))
        assert asyncio.run(_sessions(client).access_token(sid)) == "access-2"
        refresh_call = kc.requests[-1]
        assert refresh_call["body"] == {
            "grant_type": "refresh_token", "refresh_token": "refresh-1",
            "resource": MCP_URL, "client_id": CLIENT_ID,
        }
        assert store.data[sid].refresh_token == "refresh-2"

    def test_refresh_failure_drops_session(self, client: TestClient, kc: FakeKeycloak, store: MemoryStore) -> None:  # U10
        kc.expires_in = 10
        _login(client, kc)
        sid = next(iter(_sessions(client)._live))
        kc.refresh_fails = True
        with pytest.raises(LoginRequired):
            asyncio.run(_sessions(client).access_token(sid))
        assert sid not in store.data and sid not in _sessions(client)._live
        assert client.post("/query", json={}).status_code == 401

    def test_restart_rebuilds_from_store(self, client: TestClient, kc: FakeKeycloak) -> None:  # U11
        _login(client, kc)
        _sessions(client)._live.clear()  # simulate an app restart: memory gone, cookie + store kept
        assert client.post("/query", json={}).json() == {"answer": "ok"}
        assert kc.requests[-1]["body"]["grant_type"] == "refresh_token"

    def test_concurrent_refresh_happens_once(self, client: TestClient, kc: FakeKeycloak) -> None:  # U12
        kc.expires_in = 10
        _login(client, kc)
        sid = next(iter(_sessions(client)._live))
        before = sum(r["body"].get("grant_type") == "refresh_token" for r in kc.requests)

        async def two_calls() -> list[str]:
            return list(await asyncio.gather(_sessions(client).access_token(sid), _sessions(client).access_token(sid)))

        kc.expires_in = 300
        assert asyncio.run(two_calls()) == ["access-2", "access-2"]
        after = sum(r["body"].get("grant_type") == "refresh_token" for r in kc.requests)
        assert after - before == 1
