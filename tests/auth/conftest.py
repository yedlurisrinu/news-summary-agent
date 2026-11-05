"""Fixtures shared by the auth tests: a fake Keycloak, an in-memory refresh-token store,
and a TestClient for a minimal app with ``install_auth`` wired exactly as ``api/app.py`` does."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.auth.fake_keycloak import FakeKeycloak
from tests.auth.login_flow_test import BASE, MemoryStore, _build


@pytest.fixture()
def kc() -> FakeKeycloak:
    return FakeKeycloak()


@pytest.fixture()
def store() -> MemoryStore:
    return MemoryStore()


@pytest.fixture()
def client(kc: FakeKeycloak, store: MemoryStore) -> TestClient:
    return TestClient(_build(kc, store), base_url=BASE)
