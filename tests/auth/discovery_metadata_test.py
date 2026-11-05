"""
tests/auth/discovery_metadata_test.py

Issuer discovery and the memoised authorization-server metadata, over httpx2 MockTransport:
  discover_issuer
    - reads es-mcp-server's protected-resource metadata (RFC 9728) at the right URL
    - rejects a resource or authorization server that doesn't match the app's config
  AuthServerMetadata
    - lazy: nothing fetched until first use; one fetch even under concurrency
    - issuer mismatch, missing required keys and HTTP errors raise (and aren't cached)
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx2
import pytest

from auth import discovery
from auth.auth_server_metadata import AuthServerMetadata
from tests.auth.fake_keycloak import ISSUER, MCP_URL, FakeKeycloak

pytestmark = pytest.mark.anyio

PRM_URL = "http://es-mcp-lb:9599/.well-known/oauth-protected-resource/mcp"


def _patch_prm(monkeypatch: pytest.MonkeyPatch, body: dict[str, Any], seen: list[str]) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(str(request.url))
        return httpx2.Response(200, json=body)

    monkeypatch.setattr(
        discovery, "AsyncClient", lambda: httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    )


class TestDiscoverIssuer:
    async def test_returns_issuer_from_protected_resource_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[str] = []
        _patch_prm(monkeypatch, {"resource": MCP_URL, "authorization_servers": [ISSUER]}, seen)
        assert await discovery.discover_issuer() == ISSUER
        assert seen == [PRM_URL]

    @pytest.mark.parametrize(
        "body",
        [
            {"resource": "http://evil/mcp", "authorization_servers": [ISSUER]},
            {"resource": MCP_URL, "authorization_servers": ["http://evil/realms/x"]},
            {"resource": MCP_URL, "authorization_servers": []},
        ],
        ids=["resource_mismatch", "issuer_not_allowed", "no_issuer"],
    )
    async def test_mismatch_raises(self, monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]) -> None:
        _patch_prm(monkeypatch, body, [])
        with pytest.raises(ValueError):
            await discovery.discover_issuer()


class CountingDiscover:
    def __init__(self, issuer: str = ISSUER) -> None:
        self.issuer = issuer
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        await asyncio.sleep(0)  # let concurrent callers pile up on the lock
        return self.issuer


def _metadata_requests(kc: FakeKeycloak) -> int:
    return sum(r["url"].endswith("/.well-known/openid-configuration") for r in kc.requests)


class TestAuthServerMetadata:
    async def test_lazy_and_memoised(self) -> None:
        kc, discover = FakeKeycloak(), CountingDiscover()
        meta = AuthServerMetadata(discover, http_transport=kc.transport)
        assert discover.calls == 0 and kc.requests == []
        assert await meta.issuer() == ISSUER
        assert (await meta.token_endpoint()).endswith("/protocol/openid-connect/token")
        assert discover.calls == 1 and _metadata_requests(kc) == 1

    async def test_single_fetch_under_concurrency(self) -> None:
        kc, discover = FakeKeycloak(), CountingDiscover()
        meta = AuthServerMetadata(discover, http_transport=kc.transport)
        results = await asyncio.gather(*(meta.get() for _ in range(20)))
        assert all(r is results[0] for r in results)
        assert discover.calls == 1 and _metadata_requests(kc) == 1

    async def test_issuer_mismatch_raises_and_is_not_cached(self) -> None:
        kc = FakeKeycloak()

        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(200, json={**kc.metadata(), "issuer": "http://keycloak:8180/realms/other"})

        meta = AuthServerMetadata(CountingDiscover(), http_transport=httpx2.MockTransport(handler))
        with pytest.raises(ValueError, match="not same as issuer"):
            await meta.get()
        assert meta._meta is None

    async def test_missing_required_key_raises(self) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(200, json={"issuer": ISSUER})  # no endpoints

        meta = AuthServerMetadata(CountingDiscover(), http_transport=httpx2.MockTransport(handler))
        with pytest.raises(ValueError, match="missing"):
            await meta.get()

    async def test_http_error_raises_and_next_call_retries(self) -> None:
        kc, statuses = FakeKeycloak(), [503]

        def handler(request: httpx2.Request) -> httpx2.Response:
            if statuses:
                return httpx2.Response(statuses.pop())
            return httpx2.Response(200, json=kc.metadata())

        discover = CountingDiscover()
        meta = AuthServerMetadata(discover, http_transport=httpx2.MockTransport(handler))
        with pytest.raises(httpx2.HTTPStatusError):
            await meta.get()
        assert (await meta.get())["issuer"] == ISSUER
        assert discover.calls == 2
