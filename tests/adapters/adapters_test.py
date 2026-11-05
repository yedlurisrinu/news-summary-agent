"""
tests/adapters/adapters_test.py

Adapters without live services:
  - VaultRefreshTokenStore over a patched hvac Client (KV v2 under APP_VAULT_SESSIONS_PATH)
  - postgres_pool(): an unopened AsyncConnectionPool with the checkpointer's settings
  - lifespan(): opens the pool and builds the graph; never connects the MCP client
    (startup must not depend on es-mcp-server or Keycloak)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hvac.exceptions import InvalidPath

from auth.ports import StoredSession

pytestmark = pytest.mark.anyio


@pytest.fixture()
def hvac_client():
    with patch("adapters.vault_session_store.Client") as cls:
        client = cls.return_value
        client.is_authenticated.return_value = True
        yield client


class TestVaultRefreshTokenStore:
    def _store(self):
        from adapters.vault_session_store import VaultRefreshTokenStore

        return VaultRefreshTokenStore(
            "http://vault-server:8200", "token", "NEWS-SUMMARY-APP/SESSIONS"
        )

    def test_unauthenticated_token_fails_fast(self, hvac_client) -> None:
        hvac_client.is_authenticated.return_value = False
        with pytest.raises(ValueError, match="Vault authentication failed"):
            self._store()

    def test_empty_path_fails_fast(self, hvac_client) -> None:
        from adapters.vault_session_store import VaultRefreshTokenStore

        with pytest.raises(KeyError):
            VaultRefreshTokenStore("http://vault-server:8200", "token", "")

    async def test_save_get_delete_use_sid_path(self, hvac_client) -> None:
        kv = hvac_client.secrets.kv.v2
        kv.read_secret_version.return_value = {
            "data": {"data": {"refresh_token": "rt", "sub": "u1"}}
        }
        store = self._store()

        await store.save("sid1", StoredSession(refresh_token="rt", sub="u1"))
        assert await store.get("sid1") == StoredSession(refresh_token="rt", sub="u1")
        await store.delete("sid1")

        path = "NEWS-SUMMARY-APP/SESSIONS/sid1"
        kv.create_or_update_secret.assert_called_once_with(
            path=path, secret={"refresh_token": "rt", "sub": "u1"}, mount_point="secret"
        )
        assert kv.read_secret_version.call_args.kwargs["path"] == path
        kv.delete_metadata_and_all_versions.assert_called_once_with(
            path=path, mount_point="secret"
        )

    async def test_sweep_deletes_only_entries_older_than_max_age(
        self, hvac_client
    ) -> None:
        from datetime import datetime, timedelta

        kv = hvac_client.secrets.kv.v2
        kv.list_secrets.return_value = {"data": {"keys": ["old", "fresh", "folder/"]}}

        def stamp(hours_ago: float) -> str:
            t = datetime.now(UTC) - timedelta(hours=hours_ago)
            return t.strftime("%Y-%m-%dT%H:%M:%S.123456789Z")

        kv.read_secret_metadata.side_effect = lambda path, mount_point: {
            "data": {"updated_time": stamp(12 if path.endswith("/old") else 1)}
        }

        assert await self._store().sweep(39600) == 1
        kv.delete_metadata_and_all_versions.assert_called_once_with(
            path="NEWS-SUMMARY-APP/SESSIONS/old", mount_point="secret"
        )

    async def test_sweep_with_no_sessions_is_zero(self, hvac_client) -> None:
        hvac_client.secrets.kv.v2.list_secrets.side_effect = InvalidPath("no keys")
        assert await self._store().sweep(39600) == 0

    async def test_get_unknown_sid_is_none(self, hvac_client) -> None:
        hvac_client.secrets.kv.v2.read_secret_version.side_effect = InvalidPath(
            "not found"
        )
        assert await self._store().get("nope") is None


class TestPostgresPool:
    async def test_unopened_pool_with_checkpointer_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from psycopg.rows import dict_row

        from adapters.postgres import postgres_pool

        for key, value in {
            "POSTGRES_DB": "news",
            "POSTGRES_USER": "app",
            "POSTGRES_PASSWORD": "pw",
        }.items():
            monkeypatch.setenv(key, value)
        pool = postgres_pool()
        assert pool.closed  # opened by the lifespan, not here
        assert (pool.min_size, pool.max_size) == (1, 10)
        assert pool.kwargs["host"] == "postgres" and pool.kwargs["port"] == 5432
        assert (
            pool.kwargs["autocommit"] is True and pool.kwargs["prepare_threshold"] == 0
        )
        assert pool.kwargs["row_factory"] is dict_row


class TestLifespan:
    async def test_opens_pool_builds_graph_without_connecting_mcp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import adapters.lifespan as lifespan_mod

        events: list[str] = []

        @asynccontextmanager
        async def fake_pool() -> Any:
            events.append("pool open")
            yield "pool"
            events.append("pool closed")

        mcp_client = MagicMock()
        mcp_client.__aenter__ = AsyncMock(side_effect=lambda: events.append("mcp open"))
        mcp_client.__aexit__ = AsyncMock(
            side_effect=lambda *a: events.append("mcp closed")
        )
        saver = MagicMock(setup=AsyncMock())
        build = AsyncMock(return_value="compiled-graph")

        monkeypatch.setenv("SERVICE_CLIENT_SECRET", "svc-secret")
        monkeypatch.setattr(lifespan_mod, "postgres_pool", fake_pool)
        client_cls = MagicMock(return_value=mcp_client)
        monkeypatch.setattr(lifespan_mod, "Client", client_cls)
        monkeypatch.setattr(
            lifespan_mod, "AsyncPostgresSaver", MagicMock(return_value=saver)
        )
        monkeypatch.setattr(lifespan_mod, "build", build)
        get_chat_model = MagicMock(return_value="llm")
        monkeypatch.setattr(lifespan_mod, "get_chat_model", get_chat_model)

        app = MagicMock()
        app.state.session_store = None  # sweeper covered in its own tests
        async with lifespan_mod.lifespan(app):
            assert app.state.graph == "compiled-graph"
            assert events == ["pool open"]

        assert events == [
            "pool open",
            "pool closed",
        ]  # the MCP client was never entered
        lifespan_mod.AsyncPostgresSaver.assert_called_once_with("pool")
        saver.setup.assert_awaited_once()
        assert (
            client_cls.call_args.kwargs["mode"] == "2026-07-28"
        )  # version pinned: no discover probe
        (built_saver, built_llm, session), _ = build.call_args
        assert (built_saver, built_llm) == (saver, "llm")
        assert (
            isinstance(session, lifespan_mod.LazyMcpSession)
            and session._client is mcp_client
        )
        get_chat_model.assert_called_once_with("openai:gpt-4o-mini", 0.0, 10)


class TestSessionSweeper:
    async def test_sweeps_at_start_and_keeps_running_after_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        import adapters.lifespan as lifespan_mod

        store = MagicMock()
        store.sweep = AsyncMock(side_effect=[RuntimeError("vault down"), 3, 0])
        task = asyncio.create_task(lifespan_mod._sweep_sessions(store, 39600, 0))
        for _ in range(20):
            await asyncio.sleep(0)
            if store.sweep.await_count >= 3:
                break
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (
            store.sweep.await_count >= 3
        )  # the failed first sweep did not stop the loop
        store.sweep.assert_awaited_with(39600)

    async def test_lifespan_starts_and_cancels_the_sweeper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import adapters.lifespan as lifespan_mod

        @asynccontextmanager
        async def fake_pool() -> Any:
            yield "pool"

        monkeypatch.setenv("SERVICE_CLIENT_SECRET", "svc-secret")
        monkeypatch.setattr(lifespan_mod, "postgres_pool", fake_pool)
        monkeypatch.setattr(lifespan_mod, "Client", MagicMock())
        monkeypatch.setattr(
            lifespan_mod,
            "AsyncPostgresSaver",
            MagicMock(return_value=MagicMock(setup=AsyncMock())),
        )
        monkeypatch.setattr(lifespan_mod, "build", AsyncMock(return_value="graph"))
        monkeypatch.setattr(lifespan_mod, "get_chat_model", MagicMock())
        store = MagicMock(sweep=AsyncMock(return_value=0))
        app = MagicMock()
        app.state.session_store = store

        async with lifespan_mod.lifespan(app):
            import asyncio

            await asyncio.sleep(0)
            store.sweep.assert_awaited_once_with(39600)  # first sweep at start-up
        # leaving the lifespan cancelled the task cleanly (no exception escaped)
