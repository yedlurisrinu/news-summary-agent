"""
tests/config/application_config_test.py

Tests for config/application_config.py:
  - get_config returns loaded values; raises KeyError for unknown keys
  - load_config handles a missing file gracefully (logs, does not raise)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestGetConfig:
    def test_returns_loaded_value(self):
        from config.application_config import get_config

        assert get_config("MCP_URL") == "http://es-mcp-lb:9599/mcp"

    def test_unknown_key_raises(self):
        from config.application_config import get_config

        with pytest.raises(KeyError):
            get_config("NOPE_DOES_NOT_EXIST")


class TestLoadConfig:
    def test_missing_file_is_swallowed(self):
        from config import application_config

        with patch("builtins.open", side_effect=FileNotFoundError):
            application_config.load_config()  # must not raise

        # config from the autouse session load is still intact
        assert application_config.get_config("MCP_URL") == "http://es-mcp-lb:9599/mcp"
