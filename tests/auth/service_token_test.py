"""
tests/auth/service_token_test.py

Service-account auth to es-mcp-server, over an httpx2 MockTransport (no Keycloak):
  ServiceAccountTokenProvider
    - token cached while more than SERVICE_TOKEN_LEEWAY_SEC (60s) of life remains
    - refetched once it is inside the leeway, or when forced
    - one fetch for concurrent callers; a failed fetch is ServiceTokenUnavailable
  ServiceTokenAuth
    - attaches the bearer token; on 401 retries exactly once with a forced refetch
"""

from __future__ import annotations

import asyncio

import httpx2
import pytest

from auth.service_account_token import (
    ServiceAccountTokenProvider,
    ServiceTokenUnavailable,
)
from auth.service_token_auth import ServiceTokenAuth

pytestmark = pytest.mark.anyio

TOKEN_URL = "http://keycloak:8180/realms/es-mcp/protocol/openid-connect/token"


class TokenEndpoint:
    """Fake Keycloak token endpoint: hands out tok-1, tok-2, ... with a fixed lifetime."""

    def __init__(self, expires_in: int = 300, status: int = 200) -> None:
        self.expires_in = expires_in
        self.status = status
        self.requests: list[httpx2.Request] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx2.Response(self.status, json={"error": "invalid_client"})
        return httpx2.Response(
            200,
            json={
                "access_token": f"tok-{len(self.requests)}",
                "token_type": "Bearer",
                "expires_in": self.expires_in,
            },
        )


def _provider(endpoint: TokenEndpoint) -> ServiceAccountTokenProvider:
    async def token_endpoint() -> str:
        return TOKEN_URL

    return ServiceAccountTokenProvider(
        "news-summary-app-svc",
        "client-secret-for-tests",
        token_endpoint=token_endpoint,
        http_transport=httpx2.MockTransport(endpoint.handler),
    )


class TestServiceAccountTokenProvider:
    async def test_token_cached_within_leeway(self) -> None:
        endpoint = TokenEndpoint(expires_in=300)
        provider = _provider(endpoint)
        assert await provider.token() == "tok-1"
        assert await provider.token() == "tok-1"
        assert len(endpoint.requests) == 1

    async def test_client_credentials_grant_with_basic_auth(self) -> None:
        endpoint = TokenEndpoint()
        await _provider(endpoint).token()
        (request,) = endpoint.requests
        assert str(request.url) == TOKEN_URL
        assert b"grant_type=client_credentials" in request.content
        assert request.headers["authorization"].startswith("Basic ")

    async def test_refetched_once_inside_leeway(self) -> None:
        endpoint = TokenEndpoint(expires_in=30)  # less than the 60s leeway: always "expiring"
        provider = _provider(endpoint)
        assert await provider.token() == "tok-1"
        assert await provider.token() == "tok-2"

    async def test_refetched_after_expiry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import auth.service_account_token as sat

        endpoint = TokenEndpoint(expires_in=300)
        provider = _provider(endpoint)
        await provider.token()
        real_time = sat.time.time
        monkeypatch.setattr(sat.time, "time", lambda: real_time() + 301)
        assert await provider.token() == "tok-2"

    async def test_force_refetches_even_when_valid(self) -> None:
        endpoint = TokenEndpoint()
        provider = _provider(endpoint)
        await provider.token()
        assert await provider.token(force=True) == "tok-2"

    async def test_concurrent_callers_share_one_fetch(self) -> None:
        endpoint = TokenEndpoint()
        provider = _provider(endpoint)
        tokens = await asyncio.gather(*(provider.token() for _ in range(10)))
        assert set(tokens) == {"tok-1"} and len(endpoint.requests) == 1

    async def test_fetch_failure_is_service_token_unavailable(self) -> None:
        provider = _provider(TokenEndpoint(status=401))
        with pytest.raises(ServiceTokenUnavailable) as exc:
            await provider.token()
        assert exc.value.__cause__ is not None


class FakeProvider:
    def __init__(self) -> None:
        self.calls: list[bool] = []

    async def token(self, force: bool = False) -> str:
        self.calls.append(force)
        return f"svc-{len(self.calls)}"


def _mcp_client(provider: FakeProvider, statuses: list[int], seen: list[str]) -> httpx2.AsyncClient:
    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request.headers["authorization"])
        return httpx2.Response(statuses.pop(0))

    return httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler),
        auth=ServiceTokenAuth(provider),  # type: ignore[arg-type]
    )


class TestServiceTokenAuth:
    async def test_attaches_bearer_token(self) -> None:
        provider, seen = FakeProvider(), []
        async with _mcp_client(provider, [200], seen) as http:
            assert (await http.post("http://es-mcp-lb:9599/mcp")).status_code == 200
        assert seen == ["Bearer svc-1"] and provider.calls == [False]

    async def test_401_retries_once_with_forced_token(self) -> None:
        provider, seen = FakeProvider(), []
        async with _mcp_client(provider, [401, 200], seen) as http:
            assert (await http.post("http://es-mcp-lb:9599/mcp")).status_code == 200
        assert seen == ["Bearer svc-1", "Bearer svc-2"] and provider.calls == [False, True]

    async def test_second_401_is_returned_without_more_retries(self) -> None:
        provider, seen = FakeProvider(), []
        async with _mcp_client(provider, [401, 401], seen) as http:
            assert (await http.post("http://es-mcp-lb:9599/mcp")).status_code == 401
        assert len(seen) == 2 and provider.calls == [False, True]

    @pytest.mark.parametrize("status", [403, 500])
    async def test_other_errors_are_not_retried(self, status: int) -> None:
        provider, seen = FakeProvider(), []
        async with _mcp_client(provider, [status], seen) as http:
            assert (await http.post("http://es-mcp-lb:9599/mcp")).status_code == status
        assert len(seen) == 1

