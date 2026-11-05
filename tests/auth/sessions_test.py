"""
tests/auth/sessions_test.py

SessionStore paths the login-flow tests don't reach, with an in-memory store and a
scripted refresher (no Keycloak, no Vault):
  - a token response without a refresh token can't start a session
  - restart rebuild: rejected refresh drops the session; concurrent rebuilds refresh once
  - the refresh token vanishing from the store is a logout, not an outage
  - subject() for live and unknown sessions
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from auth.ports import StoredSession
from auth.sessions import AuthUnavailable, LoginRequired, SessionStore
from tests.auth.login_flow_test import MemoryStore

pytestmark = pytest.mark.anyio


class Rejected(Exception):
    error = "invalid_grant"


class Refresher:
    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail
        self.calls = 0

    async def __call__(self, refresh_token: str) -> dict[str, Any]:
        self.calls += 1
        await asyncio.sleep(0)
        if self.fail is not None:
            raise self.fail
        return {
            "access_token": f"at-{self.calls}",
            "expires_in": 300,
            "refresh_token": "rt-rotated",
        }


TOKEN = {
    "access_token": "at-0",
    "expires_in": 300,
    "refresh_token": "rt-0",
    "userinfo": {"sub": "u1"},
}


async def test_create_without_refresh_token_is_login_required() -> None:
    sessions = SessionStore(MemoryStore(), Refresher())
    with pytest.raises(LoginRequired):
        await sessions.create({"access_token": "at", "expires_in": 300})


async def test_subject_of_live_and_unknown_session() -> None:
    sessions = SessionStore(MemoryStore(), Refresher())
    sid = await sessions.create(TOKEN)
    assert sessions.subject(sid) == "u1" and sessions.subject("nope") is None


async def test_unknown_sid_is_not_active() -> None:
    assert await SessionStore(MemoryStore(), Refresher()).is_active("nope") is False


async def test_rebuild_rejected_drops_stored_session() -> None:
    store = MemoryStore()
    store.data["sid1"] = StoredSession(refresh_token="rt", sub="u1")
    sessions = SessionStore(store, Refresher(fail=Rejected()))
    assert await sessions.is_active("sid1") is False
    assert "sid1" not in store.data


async def test_rebuild_with_keycloak_down_keeps_stored_session() -> None:
    store = MemoryStore()
    store.data["sid1"] = StoredSession(refresh_token="rt", sub="u1")
    sessions = SessionStore(store, Refresher(fail=ConnectionError("keycloak down")))
    with pytest.raises(AuthUnavailable):
        await sessions.is_active("sid1")
    assert "sid1" in store.data


async def test_concurrent_rebuilds_refresh_once_and_save_rotation() -> None:
    store, refresher = MemoryStore(), Refresher()
    store.data["sid1"] = StoredSession(refresh_token="rt", sub="u1")
    sessions = SessionStore(store, refresher)
    assert all(await asyncio.gather(*(sessions.is_active("sid1") for _ in range(5))))
    assert refresher.calls == 1
    assert store.data["sid1"].refresh_token == "rt-rotated"


async def test_refresh_token_missing_from_store_is_login_required() -> None:
    store = MemoryStore()
    sessions = SessionStore(
        store, Refresher(), refresh_skew_seconds=10_000
    )  # always "near expiry"
    sid = await sessions.create(TOKEN)
    store.data.clear()  # the stored entry is gone
    with pytest.raises(LoginRequired):
        await sessions.access_token(sid)
    assert sessions.subject(sid) is None


async def test_guard_outside_session_middleware_is_a_wiring_error() -> None:
    from auth.guard import AuthGuard

    async def app(scope: Any, receive: Any, send: Any) -> None:
        raise AssertionError("must not be reached")

    guard = AuthGuard(app, sessions=SessionStore(MemoryStore(), Refresher()))
    with pytest.raises(RuntimeError, match="SessionMiddleware"):
        await guard({"type": "http", "path": "/chat"}, AsyncMock(), AsyncMock())
