"""Async YAML / JSON readers; paths are relative to the evals folder (BASE_PATH).

The file read runs in a worker thread so it never blocks the event loop.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import yaml

from evals.e2e.helpers.constants import E2E_BASE_PATH


async def _read_text(relative_path: str | Path) -> str:
    """Read E2E_BASE_PATH/relative_path as UTF-8 text in a worker thread."""
    return await asyncio.to_thread(
        E2E_BASE_PATH.joinpath(relative_path).read_text, encoding="utf-8"
    )


async def yaml_loader(relative_path: str | Path) -> Any:
    """Parse a YAML file under the evals folder. FileNotFoundError if it is missing."""
    return yaml.safe_load(await _read_text(relative_path))


async def json_loader(relative_path: str | Path) -> Any:
    """Parse a JSON file under the evals folder. FileNotFoundError if it is missing."""
    return json.loads(await _read_text(relative_path))
