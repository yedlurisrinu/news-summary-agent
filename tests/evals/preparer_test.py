"""
tests/evals/preparer_test.py

Building judge inputs (evals/judge/preparer.py) from the frozen captures of run 20261112T205532Z:
  - a missing capture is skipped with a warning, not a crash
  - judged_cases drops `judge: skip` cases
  - to_judge_input keeps only the criteria a case lists and skips fixed replies (no articles)
"""

from __future__ import annotations

import logging

import pytest

from evals.eval_config.load_eval_config import load_config
from evals.judge.preparer import (
    get_rubric,
    judged_cases,
    load_e2e_fixture_captures,
    to_judge_input,
)

pytestmark = pytest.mark.anyio

RUN_ID = "20261112T205532Z"


@pytest.fixture(autouse=True)
def _eval_config() -> None:
    load_config()


def _case(case_id: str, judge: object = None) -> dict[str, object]:
    return {
        "id": case_id,
        "query": "q",
        "judge": judge if judge is not None else {"criteria": ["cr1", "cr2"]},
    }


async def test_missing_capture_is_skipped_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        captures = await load_e2e_fixture_captures(
            RUN_ID, [_case("E2E-001"), _case("E2E-999")]
        )
    assert list(captures) == ["E2E-001"]
    assert "E2E-999" in caplog.text


async def test_judged_cases_drops_skip() -> None:
    cases = [_case("E2E-001"), _case("E2E-018", "skip")]
    assert [c["id"] for c in await judged_cases(cases)] == ["E2E-001"]


async def test_to_judge_input_filters() -> None:
    cases = [
        _case("E2E-001"),  # summary with articles, both criteria
        _case("E2E-002", {"criteria": ["cr1"]}),  # only cr1 listed
        _case("E2E-015"),  # judged in the yaml, but captured a fixed no-results reply
        _case("E2E-018", "skip"),
    ]
    captures = await load_e2e_fixture_captures(RUN_ID, cases)
    items = await to_judge_input(cases, captures, await get_rubric())
    assert [(i.case_id, i.criterion.id) for i in items] == [
        ("E2E-001", "cr1"),
        ("E2E-001", "cr2"),
        ("E2E-002", "cr1"),
    ]
