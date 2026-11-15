"""Eval data models: cases, criteria, verdicts and judge-vs-label calibration rows."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Kind = Literal["deterministic", "judged"]


@dataclass(frozen=True)
class Case:
    """A hand-written judge case (query, article texts, summary, human verdict per criterion)."""

    id: str
    case_query: str
    articles: tuple[str, ...]
    summary: str
    human: dict[str, bool]


@dataclass(frozen=True)
class Criterion:
    """One rubric criterion: its id (cr1, cr2), kind and the rule text the judge sees."""

    id: str
    kind: Kind  # from kind above
    rule_text: str  # The rule text llm judge sees


class CaseVerdict(BaseModel):
    """
    This holds the result of the judge's answer for per case per criterion
    """

    model_config = ConfigDict(frozen=True)
    evidence: str | None = Field(
        description="Quote the summary sentence or article text the case's criterion verdict rest on "
    )
    passed: bool | None = Field(
        description="indicating if the case passed or failed at the criterion"
    )

    case_id: str = Field(description=" case id on which criterion verdict received ")

    criterion_id: str = Field(
        description=" the id of the criterion used for this verdict "
    )
    judge_config_version: str = Field(
        description=" the judge version that produced the verdict "
    )
    run: int = Field(description=" current run ")
    error: str | None = Field(description=" If the call failed for any reason ")


class CriterionVerdict(BaseModel):
    """
    This holds the result of the judge's criterion verdict for a case
    """

    evidence: str = Field(
        description="Quote the summary sentence or article text the case's criterion verdict rest on "
    )
    passed: bool = Field(
        description="indicating if the case passed or failed at the criterion"
    )


class Calibration(BaseModel):
    """One judge vs the human labels, for one criterion or for all criteria pooled ("all")."""

    criterion_id: (
        str  # rubric criterion id (cr1, cr2), or "all" for every criterion pooled
    )
    total_items: int
    agree: int  # both human and judge agree on either pass or fail
    false_pass: int  # judge passed a case that was marked as failed by human
    false_fail: int  # judge failed a case that human marked as passed
    human_fails: int  # total human marked as failed
    human_passes: int  # total human marked as passes
