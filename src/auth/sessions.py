"""Server-side sessions (decision S881).

- Access token + expiry + ``sub``: in process memory, keyed by ``sid``.
- Refresh token: in the ``RefreshTokenStore`` (Vault), keyed by ``sid`` - so after an app
  restart a valid cookie is enough to rebuild the session silently.
- The browser only ever holds ``sid``, inside the signed session cookie.

Lifecycle ("stays logged in while the user interacts", L75):
  valid access token -> use it
  expiring/expired   -> refresh at Keycloak (works while the Keycloak SSO session lives)
   refresh rejected   -> (Keycloak says invalid_grant: SSO session idle/max expired,
                        logged out, revoked) drop the session -> LoginRequired -> login
   no answer          -> (Keycloak down, network error, 5xx, other OAuth error, session
                       store down) KEEP the session -> AuthUnavailable -> 503 (F-E2E6-3)
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from auth.ports import RefreshTokenStore, StoredSession

logger = logging.getLogger(__name__)

Refresher = Callable[[str], Awaitable[dict[str, Any]]]
"""``async (refresh_token) -> token response``; ``KeycloakOAuth.refresh`` in production."""


class LoginRequired(Exception):
    """The session is gone or Keycloak rejected its refresh token: the user must log in again."""

class AuthUnavailable(Exception):
    """ Keycloak or the session store gave no usable answer; the session is kept 
    """
def _is_rejection(exc: Exception) ->bool:
    """True only when Keycloak answered and said the refresh token is no longer valid.
Authlib raises ``OAuthError`` with ``.error == "invalid_grant"`` for that 400 response.
Duck-typed on ``.error`` so this module stays free of Authlib (the Refresher is a port).
Anything else - transport errors, 5xx (``HTTPStatusError``), other OAuth errors such as
``invalid_client`` (a configuration problem a new login would not fix) - is not a
statement about this session, so it must not delete it.
"""
    return getattr(exc,"error", None) == "invalid_grant"

@dataclass
class _LiveSession:
    access_token: str
    expires_at: float  # epoch seconds
    sub: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

def _expires_at(token: Mapping[str, Any]) -> float:
    return time.time() + float(token.get("expires_in", 0))

class SessionStore:
    def __init__(
        self,
        store: RefreshTokenStore,
        refresher: Refresher,
        refresh_skew_seconds: int = 30,
    ) -> None:
        self._store = store
        self._refresher = refresher
        self._skew = refresh_skew_seconds
        self._live: dict[str, _LiveSession] = {}
        self._rebuild_locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def _tag(sid: str) -> str:
        """Short, non-secret handle for logs (never log the full sid or any token)."""
        return sid[:8]

    async def create(self, token: Mapping[str, Any]) -> str:
        """Start a session from the callback's token response; return the new ``sid``."""
        userinfo = token.get("userinfo") or {}
        sub = str(userinfo.get("sub", ""))
        refresh_token = token.get("refresh_token")
        if not refresh_token:
            raise LoginRequired("token response has no refresh_token")
        sid = secrets.token_urlsafe(32)
        await self._store.save(sid, StoredSession(refresh_token=str(refresh_token), sub=sub))
        self._live[sid] = _LiveSession(
            access_token=str(token["access_token"]), expires_at=_expires_at(token), sub=sub
        )
        logger.info("session %s created", self._tag(sid))
        return sid

    async def is_active(self, sid: str) -> bool:
        """True if the session is in memory, or can be rebuilt from the store (restart case).
        Raises ``AuthUnavailable`` (session kept) when Keycloak or the store cannot answer.
        """
        if sid in self._live:
            return True
        lock = self._rebuild_locks.setdefault(sid, asyncio.Lock())
        async with lock:
            if sid in self._live:  # another request rebuilt it while we waited
                return True
            stored = await self._get(sid)
            if stored is None:
                self._rebuild_locks.pop(sid, None)
                return False
            try:
                token = await self._refresher(stored.refresh_token)
            except Exception as exc:  # expired SSO session, logged out , revoked
                if _is_rejection(exc):  # expired SSO session, logged out, revoked
                    logger.info("session %s not rebuilt: refresh rejected", self._tag(sid))
                    await self.drop(sid)
                    return False
                logger.warning("session %s not rebuilt, Keycloak unavailable: %s", self._tag(sid), type(exc).__name__)
                raise AuthUnavailable("Keycloak unavailable") from exc

            self._live[sid] = _LiveSession(
                access_token=str(token["access_token"]),
                expires_at=_expires_at(token),
                sub=stored.sub,
            )
            await self._save_rotated(sid, stored, token)
            logger.info("session %s rebuilt from the store", self._tag(sid))
            return True

    async def access_token(self, sid: str) -> str:
        """A fresh access token, refreshing it at Keycloak first when it is near expiry.
        Raises ``LoginRequired`` (session dropped) or ``AuthUnavailable`` (session kept).
        """
        live = self._live.get(sid)
        if live is None:
            if not await self.is_active(sid):
                raise LoginRequired("no session")
            live = self._live[sid]
        async with live.lock:  # one refresh per session, even with concurrent requests
            if live.expires_at - time.time() > self._skew:
                return live.access_token
            stored = await self._store.get(sid)
            if stored is None:
                await self.drop(sid)
                raise LoginRequired("refresh token missing")
            try:
                token = await self._refresher(stored.refresh_token)
            except Exception as exc:
                if _is_rejection(exc):
                    logger.info("session %s refresh failed: refresh rejected", self._tag(sid))
                    await self.drop(sid)
                    raise LoginRequired("refresh rejected") from exc
                logger.warning("session %s refresh unavailable, session kept: %s", self._tag(sid), type(exc).__name__)
                raise AuthUnavailable("Keycloak unavailable") from exc

            live.access_token = str(token["access_token"])
            live.expires_at = _expires_at(token)
            await self._save_rotated(sid, stored, token)
            logger.info("session %s refreshed", self._tag(sid))
            return live.access_token

    async def _get(self, sid: str) -> StoredSession | None:
            """Read the refresh token; a store outage is ``AuthUnavailable``, not a logout."""
            try:
                return await self._store.get(sid)
            except Exception as exc:
                logger.warning("session %s store unavailable: %s", self._tag(sid), type(exc).__name__)
                raise AuthUnavailable("session store unavailable") from exc
            
    def subject(self, sid: str) -> str | None:
        live = self._live.get(sid)
        return live.sub if live else None

    async def drop(self, sid: str) -> None:
        """Forget the session everywhere (memory + store)."""
        self._live.pop(sid, None)
        self._rebuild_locks.pop(sid, None)
        try:
            await self._store.delete(sid)
        except Exception as exc:  # noqa: BLE001 - logged out either way; the entry is orphaned
            logger.warning("session %s not deleted from the store: %s", self._tag(sid), type(exc).__name__)

    async def _save_rotated(
        self, sid: str, stored: StoredSession, token: Mapping[str, Any]
    ) -> None:
        new_refresh = token.get("refresh_token")
        if new_refresh and new_refresh != stored.refresh_token:
            await self._store.save(sid, StoredSession(refresh_token=str(new_refresh), sub=stored.sub))
