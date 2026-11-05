"""
tests/config/logging_setup_test.py

setup_logging(): loads resources/logging_config.json, and falls back to basicConfig
(logging an error, not raising) when the file is missing or invalid.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

from config.logging_setup import setup_logging


def test_loads_dict_config_from_resources() -> None:
    with patch("config.logging_setup.logging.config.dictConfig") as dict_config:
        setup_logging()
    (config,), _ = dict_config.call_args
    assert config["version"] == 1 and "my_project" not in config.get("loggers", {})


def test_missing_file_falls_back_to_basic_config() -> None:
    with (
        patch("config.logging_setup.logging.basicConfig") as basic,
        patch("config.logging_setup.logger") as logger,
    ):
        setup_logging("does-not-exist.json")
    basic.assert_called_once_with(level=logging.INFO)
    logger.exception.assert_called_once()
