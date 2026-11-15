"""A scripted judge for tests and demos: returns fixed verdicts, makes no API calls."""

from collections.abc import Mapping

from evals.e2e.model.judge_input import JudgeInput
from evals.e2e.model.models import CaseVerdict


class FakeJudge:
    """Scripted verdicts behind the Judge protocol; no API calls. None = the judge errored."""

    def __init__(
        self,
        script: Mapping[tuple[str, str], bool | None],
        config_version: str = "fake-v1",
    ) -> None:
        """script maps (case_id, criterion_id) → passed; None makes that verdict an error."""
        self.script, self.config_version = script, config_version

    def __call__(self, items: list[JudgeInput], run: int = 0) -> list[CaseVerdict]:
        """Return the scripted verdict for every item, in input order. KeyError if an item is not scripted."""
        out = []
        for i in items:
            passed = self.script[(i.case_id, i.criterion.id)]
            out.append(
                CaseVerdict(
                    evidence="scripted",
                    passed=passed,
                    case_id=i.case_id,
                    criterion_id=i.criterion.id,
                    judge_config_version=self.config_version,
                    run=run,
                    error=None if passed is not None else "scripted_error",
                )
            )
        return out
