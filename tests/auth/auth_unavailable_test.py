"""Auth back-end outages: Keycloak or Vault down must give 503 and keep the session."""

import pytest
from fastapi.testclient import TestClient

from auth.ports import StoredSession
from tests.auth.fake_keycloak import FakeKeycloak
from tests.auth.login_flow_test import BASE, MemoryStore, _build, _login, _sessions


class FlakyStore(MemoryStore):
    """A MemoryStore that can simulate a Vault outage on reads (``get``) and/or deletes."""

    def __init__(self) -> None:
        super().__init__()
        self.load_down = False
        self.delete_down = False

    async def get(self, sid: str) -> StoredSession | None:
        if self.load_down:
            raise ConnectionError("vault down")
        return await super().get(sid)

    async def delete(self, sid: str) -> None:
        if self.delete_down:
            raise ConnectionError("vault down")
        await super().delete(sid)


class TestAuthUnavailable:  # an outage is not a logout
    @pytest.mark.parametrize("outage", ["http_503", "connect", "invalid_client"])
    def test_keycloak_outage_is_503_and_keeps_session(
        self, client: TestClient, kc: FakeKeycloak, store: MemoryStore, outage: str
    ) -> None:
        kc.expires_in = 10
        _login(client, kc)
        sid = next(iter(_sessions(client)._live))
        kc.refresh_outage = outage
        r = client.get("/chat", headers={"accept": "text/html"}, follow_redirects=False)
        assert r.status_code == 503 and "temporarily unavailable" in r.text
        assert (
            r.headers["retry-after"] == "30"
            and r.headers["cache-control"] == "no-store"
        )
        r = client.post("/query", json={})
        assert r.status_code == 503 and r.json() == {"error": "auth_unavailable"}
        assert sid in store.data and sid in _sessions(client)._live  # nothing deleted
        kc.refresh_outage, kc.expires_in = None, 300  # Keycloak back: same cookie works
        assert client.post("/query", json={}).json() == {"answer": "ok"}

    def test_restart_with_keycloak_down_keeps_store_entry(
        self, client: TestClient, kc: FakeKeycloak, store: MemoryStore
    ) -> None:
        _login(client, kc)
        sid = next(iter(_sessions(client)._live))
        _sessions(client)._live.clear()  # app restart
        kc.refresh_outage = "connect"
        assert client.post("/query", json={}).status_code == 503
        assert sid in store.data
        kc.refresh_outage = None
        assert client.post("/query", json={}).json() == {"answer": "ok"}

    def test_vault_outage_is_503_and_keeps_session(
        self, kc: FakeKeycloak
    ) -> None:
        flaky = FlakyStore()
        client = TestClient(_build(kc, flaky), base_url=BASE)
        kc.expires_in = 10
        _login(client, kc)
        sid = next(iter(_sessions(client)._live))
        flaky.load_down = True
        assert client.post("/query", json={}).status_code == 503
        assert sid in _sessions(client)._live
        flaky.load_down, kc.expires_in = False, 300
        assert client.post("/query", json={}).json() == {"answer": "ok"}

    def test_rejection_with_vault_down_still_logs_out(
        self, kc: FakeKeycloak
    ) -> None:
        flaky = FlakyStore()
        client = TestClient(_build(kc, flaky), base_url=BASE)
        kc.expires_in = 10
        _login(client, kc)
        sid = next(iter(_sessions(client)._live))
        kc.refresh_fails = True
        flaky.delete_down = (
            True  # Keycloak rejects, and the Vault delete hits an outage
        )
        r = client.get("/chat", headers={"accept": "text/html"}, follow_redirects=False)
        assert r.status_code == 302 and r.headers["location"] == "/login"
        assert sid not in _sessions(client)._live
