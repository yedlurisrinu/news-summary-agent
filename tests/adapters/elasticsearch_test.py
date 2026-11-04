"""
tests/adapters/elasticsearch_test.py

Tests for adapters/elasticsearch.py get_es_client():
  - missing API key raises RuntimeError
  - ping() returning False raises RuntimeError
  - success returns the client; lru_cache yields a singleton
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _clear_cache():
    from adapters.elasticsearch import get_es_client

    get_es_client.cache_clear()


class TestGetEsClient:
    def test_missing_api_key_raises(self, monkeypatch):
        _clear_cache()
        monkeypatch.delenv("ELASTIC_READ_API_KEY", raising=False)
        from adapters.elasticsearch import get_es_client

        with pytest.raises(RuntimeError, match="ELASTIC_URL and ELASTIC_READ_API_KEY"):
            get_es_client()

    def test_ping_false_raises(self, monkeypatch):
        _clear_cache()
        monkeypatch.setenv("ELASTIC_READ_API_KEY", "key")
        instance = MagicMock()
        instance.ping.return_value = False
        with patch("adapters.elasticsearch.Elasticsearch", return_value=instance):
            from adapters.elasticsearch import get_es_client

            with pytest.raises(RuntimeError, match="ping returned False"):
                get_es_client()

    def test_success_returns_singleton(self, monkeypatch):
        _clear_cache()
        monkeypatch.setenv("ELASTIC_READ_API_KEY", "key")
        instance = MagicMock()
        instance.ping.return_value = True
        with patch(
            "adapters.elasticsearch.Elasticsearch", return_value=instance
        ) as ctor:
            from adapters.elasticsearch import get_es_client

            first = get_es_client()
            second = get_es_client()

        assert first is instance
        assert first is second
        ctor.assert_called_once()

    def teardown_method(self):
        _clear_cache()
