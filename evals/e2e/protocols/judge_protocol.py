"""The interface every judge (NewsJudge, FakeJudge) implements."""

from typing import Protocol

from evals.e2e.model.judge_input import JudgeInput
from evals.e2e.model.models import CaseVerdict


class Judge(Protocol):
    """A judge grades (case, criterion) items; `run` tags the verdicts with the repeat number (1-based)."""

    def __call__(self, use_cases: list[JudgeInput], run: int = 0) -> list[CaseVerdict]:
        """Grade each item; return one CaseVerdict per item, in input order."""
        ...
