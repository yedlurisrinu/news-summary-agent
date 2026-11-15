"""JSON writers for eval outputs; paths are relative to the evals folder (BASE_PATH)."""

import dataclasses
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from evals.helpers.constants import BASE_PATH
from evals.model.capture import Capture
from evals.model.judge_config import JudgeConfig
from evals.model.judge_input import JudgeInput


def to_json(o: Any) -> Any:
    """Fallback for objects json can't encode: Pydantic models and dataclasses → dicts, anything else → str."""
    if isinstance(o, BaseModel):
        return o.model_dump(mode="json")
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    return str(o)


def _write(payload: Any, path: str | Path) -> None:
    """Serialise payload as indented UTF-8 JSON at BASE_PATH/path, creating parent folders."""
    new_path = BASE_PATH.joinpath(path)
    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=to_json),
        encoding="utf-8",
    )


def dump_list_of_records(items: list[Any], path: str | Path) -> None:
    """Write a list of records (pydantic models, dataclasses, plain values) as a JSON array."""
    _write(items, path)


def dump_judge_items(items: list[JudgeInput], path: str | Path) -> None:
    """Write judge inputs (JudgeInput dataclasses) as a JSON array, for debugging."""
    _write(items, path)


def dump_captures(captures: dict[str, Capture], path: str | Path) -> None:
    """Write captures as a JSON object case_id → capture fields, for debugging."""
    _write(
        {case_id: dataclasses.asdict(cap) for case_id, cap in captures.items()}, path
    )


def dump_judge_config(judge_config: JudgeConfig, path: str | Path) -> None:
    """Write a JudgeConfig as JSON (judge_config.json next to its verdicts)."""
    _write(judge_config, path)
