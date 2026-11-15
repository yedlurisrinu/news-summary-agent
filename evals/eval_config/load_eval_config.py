"""Eval config — loads evals/resources/eval_config.yaml into a module-level dict."""

import logging
from typing import Any

import yaml

from evals.helpers.constants import BASE_PATH

log = logging.getLogger(__name__)

config: dict[str, Any] = {}


def load_config() -> None:
    """Load eval_config.yaml into the module-level ``config`` dict; re-raises on failure."""
    global config
    try:
        with open(
            BASE_PATH.joinpath("resources/eval_config.yaml"), encoding="utf-8"
        ) as file_handle:
            config = yaml.safe_load(file_handle)
    except Exception:
        log.exception("Exception while loading eval configuration")
        raise


def get_config(key: str) -> Any:
    """Return a config value by key (str, number or list). Raises ``KeyError`` if absent."""
    return config[key]
