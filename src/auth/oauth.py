"""Authlib wiring for Keycloak, registered lazily after discovery.

Why lazy: registering the login client needs the authorization-server metadata, which
needs es-mcp-server (issuer discovery) and Keycloak. It happens on the first
``/auth/login``, and a failed attempt is retried on the next one. App startup needs
neither: ``LazyMcpSession`` (adapters/mcp_session.py) connects on the first tool call, and
issuer discovery happens on the first login or the first MCP call.

Verified against Authlib 1.9.1:
- ``authorize_redirect(request, redirect_uri, resource=...)`` puts ``resource``, PKCE S256,
  ``state`` and ``nonce`` on the authorize URL; the PKCE verifier/state/nonce live in the
  signed session cookie until the callback.
- ``authorize_access_token(request, resource=...)`` forwards ``resource`` to the token
  request and validates the ID token (signature via ``jwks_uri``, nonce, iss, aud).
- Authlib does NOT check the RFC 9207 ``iss`` callback parameter - ``routes`` does.
- ``AsyncOAuth2Client.refresh_token(url, refresh_token=..., resource=...)`` posts
  grant_type, refresh_token, resource and client_id, with no Authorization header.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from authlib.integrations.httpx_client import AsyncOAuth2Client
from authlib.integrations.starlette_client import OAuth

from auth.auth_server_metadata import AuthServerMetadata
from auth.config import BffConfig

logger = logging.getLogger(__name__)

CLIENT_NAME = "keycloak"


class KeycloakOAuth:
    """Owns the Authlib client for the CIMD public client."""

    def __init__(
        self,
        config: BffConfig,
        as_meta: AuthServerMetadata,
        http_transport: Any | None = None,
    ) -> None:
        self._config = config
        self._issuer: str | None = None
        self._as_meta = as_meta
        # Test seam: an httpx2 transport (e.g. MockTransport) used for every Authlib call.
        self._client_kwargs: dict[str, Any] = (
            {"transport": http_transport} if http_transport is not None else {}
        )
        self._oauth = OAuth()
        self._client: Any | None = None
        self._lock = asyncio.Lock()

    @property
    def issuer(self) -> str | None:
        """The discovered issuer, or ``None`` before the first successful discovery."""
        return self._issuer

    async def client(self) -> Any:
        """Return the registered Authlib client, discovering and registering on first use.

        A failed discovery is not cached: the next login attempt tries again.
        """
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is None:
                meta = await self._as_meta.get()
                self._issuer = meta["issuer"]
                self._client = self._oauth.register(
                    name=CLIENT_NAME,
                    client_id=self._config.client_id,
                    client_kwargs={
                        "scope": "openid",
                        "code_challenge_method": "S256",
                        "token_endpoint_auth_method": "none",  # public client (CIMD)
                        **self._client_kwargs,
                    },
                    **meta,
                )
                logger.info("OAuth client registered for issuer %s", self._issuer)
        return self._client

    async def refresh(self, refresh_token: str) -> dict[str, Any]:
        """Exchange a refresh token at Keycloak's token endpoint (never at the MCP server).

        Raises on any failure (expired SSO session -> invalid_grant, network, ...).
        """
        token_endpoint = await self._as_meta.token_endpoint()
        async with AsyncOAuth2Client(
            client_id=self._config.client_id,
            token_endpoint_auth_method="none",
            **self._client_kwargs,
        ) as http:
            token = await http.refresh_token(
                token_endpoint,
                refresh_token=refresh_token,
                resource=self._config.mcp_url,
            )
        return dict(token)
