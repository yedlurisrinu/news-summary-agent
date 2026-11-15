"""The judge configuration and its version hash."""

import hashlib
from typing import Any

from pydantic import BaseModel, ConfigDict

from evals.e2e.model.models import Criterion


class JudgeConfig(BaseModel):
    """Everything that determines a judge's verdicts. Frozen; `version` hashes all of it."""

    model_config = ConfigDict(frozen=True)
    model_name: str  # gpt-6.1-terra
    reasoning_effort: str | None = None
    timeout_s: int = 600
    max_retries: int = 1
    system_prompt: str  # JUDGE_SYSTEM text
    human_prompt: str  # JUDGE_HUMAN text
    article_separator: str
    rubric: tuple[Criterion, ...]  # normalised rule_text
    verdict_schema: dict[str, Any]  # CriterionVerdict.model_json_schema()

    @property
    def version(self) -> str:
        """First 12 hex characters of the SHA-256 of the config's JSON dump.

        Any change to the model, prompts, rubric or verdict schema gives a new version, so
        verdicts from different judge setups are never mixed.
        """
        blob = (
            self.model_dump_json()
        )  # deterministic: fixed field order, no whitespace choices
        return hashlib.sha256(blob.encode()).hexdigest()[:12]
