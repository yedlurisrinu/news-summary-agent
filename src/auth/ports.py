"""The auth package's seams: protocols that the tests replace with in-memory fakes.

``RefreshTokenStore``
    Durable home of each session's refresh token, so a restart of the app is invisible
    to the user. Implementation: Vault KV v2 under
    ``secret/NEWS-SUMMARY-APP/SESSIONS/<sid>`` using ``hvac`` (sync), called through
    ``asyncio.to_thread`` so the event loop is never blocked.

``DiscoverIssuer``
    ``async () -> str``, implemented in ``auth/discovery.py`` and taken by
    ``AuthServerMetadata``: fetch es-mcp-server's protected-resource metadata (RFC 9728),
    check ``resource == MCP_URL`` and ``authorization_servers[0] == ALLOWED_ISSUER_URL``,
    and return that issuer. Raise on any mismatch or network error.

The session/guard code (``auth/sessions.py``, ``auth/guard.py``) depends only on these
protocols, never on hvac, httpx2 or Authlib directly. Other ``auth`` modules do use
httpx2/Authlib directly: ``discovery.py``, ``auth_server_metadata.py``, ``oauth.py`` and
``service_account_token.py``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class StoredSession:
    """What survives an app restart for one session id."""

    refresh_token: str
    sub: str


class RefreshTokenStore(Protocol):
    """Durable refresh-token storage, keyed by session id (``sid``)."""

    async def get(self, sid: str) -> StoredSession | None:
        """Return the stored session, or ``None`` when nothing is stored for ``sid``."""
        ...

    async def save(self, sid: str, session: StoredSession) -> None:
        """Create or replace the stored session for ``sid``."""
        ...

    async def delete(self, sid: str) -> None:
        """Remove the stored session; a missing ``sid`` is not an error."""
        ...


DiscoverIssuer = Callable[[], Awaitable[str]]
"""``async () -> issuer``: discovery from the MCP server's protected-resource metadata."""
