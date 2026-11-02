"""A tiny in-memory Keycloak stand-in for the auth tests (served via httpx2.MockTransport).

It implements just what Authlib touches: OIDC discovery, JWKS, and the token endpoint for
authorization_code and refresh_token grants. Every request it receives is recorded so the
tests can assert on what the app sent (e.g. ``resource`` on the token and refresh calls).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs

import httpx2
from joserfc import jwt
from joserfc.jwk import RSAKey

ISSUER = "http://keycloak:8180/realms/es-mcp"
CLIENT_ID = "https://news-cimd-host/clients/news-summary-app-cimd.json"
MCP_URL = "http://es-mcp-lb:9599/mcp"


@dataclass
class FakeKeycloak:
    issuer: str = ISSUER
    client_id: str = CLIENT_ID
    nonce: str | None = None  # the test copies it from the authorize URL
    refresh_fails: bool = False
    refresh_outage: str | None = None  # "http_503" | "connect" | "invalid_client"
    rotate_refresh: bool = False
    expires_in: int = 300
    requests: list[dict[str, Any]] = field(default_factory=list)
    _key: RSAKey = field(default_factory=lambda: RSAKey.generate_key(2048, parameters={"kid": "k1"}))
    _counter: int = 0

    @property
    def transport(self) -> httpx2.MockTransport:
        return httpx2.MockTransport(self._handle)

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        url = str(request.url)
        body = {k: v[0] for k, v in parse_qs(request.content.decode()).items()} if request.content else {}
        self.requests.append({"method": request.method, "url": url, "body": body})
        if url == f"{self.issuer}/.well-known/openid-configuration":
            return httpx2.Response(200, json=self.metadata())
        if url == f"{self.issuer}/protocol/openid-connect/certs":
            return httpx2.Response(200, json={"keys": [self._key.as_dict(private=False)]})
        if url == f"{self.issuer}/protocol/openid-connect/token":
            if body.get("grant_type") == "refresh_token" and self.refresh_outage:
                return self._outage(request)
            return self._token(body)
        return httpx2.Response(404, json={"error": "not_found"})

    def metadata(self) -> dict[str, Any]:
        base = f"{self.issuer}/protocol/openid-connect"
        return {
            "issuer": self.issuer,
            "authorization_endpoint": f"{base}/auth",
            "token_endpoint": f"{base}/token",
            "jwks_uri": f"{base}/certs",
            "id_token_signing_alg_values_supported": ["RS256"],
        }

    def _outage(self, request: httpx2.Request) -> httpx2.Response:
        if self.refresh_outage == "connect":
            raise httpx2.ConnectError("connection refused", request=request)
        if self.refresh_outage == "invalid_client":
            return httpx2.Response(400, json={"error": "invalid_client"})
        return httpx2.Response(503, text="<html>Service Unavailable</html>")

    def _token(self, body: dict[str, str]) -> httpx2.Response:
        grant = body.get("grant_type")
        if grant == "refresh_token" and self.refresh_fails:
            return httpx2.Response(400, json={"error": "invalid_grant", "error_description": "Session not active"})
        self._counter += 1
        now = int(time.time())
        payload: dict[str, Any] = {
            "token_type": "Bearer",
            "access_token": f"access-{self._counter}",
            "expires_in": self.expires_in,
            "refresh_token": f"refresh-{self._counter}" if (grant != "refresh_token" or self.rotate_refresh) else body["refresh_token"],
            "scope": "openid",
        }
        if grant == "authorization_code":
            claims = {
                "iss": self.issuer, "aud": self.client_id, "sub": "user-123",
                "iat": now, "exp": now + 300, "nonce": self.nonce,
            }
            payload["id_token"] = jwt.encode({"alg": "RS256", "kid": "k1"}, claims, self._key)
        return httpx2.Response(200, json=payload)
