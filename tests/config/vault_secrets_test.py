"""
tests/config/vault_secrets_test.py

Tests for config/vault_secrets.py:
  - missing VAULT_ADDR / VAULT_TOKEN raises
  - missing VAULT_SECRET_PATH raises
  - failed Vault authentication raises
  - success path injects secrets into os.environ
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestLoadSecrets:
    def test_missing_addr_or_token_raises(self, monkeypatch):
        from config.vault_secrets import load_secrets

        monkeypatch.delenv("VAULT_ADDR", raising=False)
        monkeypatch.delenv("VAULT_TOKEN", raising=False)
        with pytest.raises(ValueError, match="VAULT_ADDR and VAULT_TOKEN"):
            load_secrets()

    def test_missing_secret_path_raises(self, monkeypatch):
        from config.vault_secrets import load_secrets

        monkeypatch.setenv("VAULT_ADDR", "http://vault:8200")
        monkeypatch.setenv("VAULT_TOKEN", "tok")
        monkeypatch.delenv("VAULT_SECRET_PATH", raising=False)
        with pytest.raises(ValueError, match="VAULT_SECRET_PATH"):
            load_secrets()

    def test_auth_failure_raises(self, monkeypatch):
        from config.vault_secrets import load_secrets

        monkeypatch.setenv("VAULT_ADDR", "http://vault:8200")
        monkeypatch.setenv("VAULT_TOKEN", "tok")
        monkeypatch.setenv("VAULT_SECRET_PATH", "secret/CREDENTIALS")

        client = MagicMock()
        client.is_authenticated.return_value = False
        with (
            patch("config.vault_secrets.Client", return_value=client),
            pytest.raises(ValueError, match="authentication failed"),
        ):
            load_secrets()

    def test_success_injects_env(self, monkeypatch):
        from config.vault_secrets import load_secrets

        monkeypatch.setenv("VAULT_ADDR", "http://vault:8200")
        monkeypatch.setenv("VAULT_TOKEN", "tok")
        monkeypatch.setenv("VAULT_SECRET_PATH", "secret/CREDENTIALS")
        monkeypatch.delenv("INJECTED_KEY", raising=False)

        client = MagicMock()
        client.is_authenticated.return_value = True
        client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {"data": {"INJECTED_KEY": "injected_value"}}
        }
        with patch("config.vault_secrets.Client", return_value=client):
            load_secrets()

        import os

        assert os.environ["INJECTED_KEY"] == "injected_value"
