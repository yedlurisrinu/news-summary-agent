"""Hand check of the agreement maths with scripted judges (no API calls).

  uv run python -m evals.helpers.fake_script

The same scenario is asserted in tests/evals/calibration_test.py.
"""

import asyncio

from evals.e2e.helpers.fake_judge import FakeJudge
from evals.e2e.judge.calibration import calibrate, calibrate_2judges, wilson
from evals.e2e.model.judge_input import JudgeInput
from evals.e2e.model.models import CaseVerdict, Criterion

P, F = True, False
items_keys = [
    ("A", "cr1"),
    ("A", "cr2"),
    ("B", "cr1"),
    ("B", "cr2"),
    ("C", "cr1"),
    ("C", "cr2"),
]

ground_truth = {
    ("A", "cr1"): P,
    ("A", "cr2"): F,
    ("B", "cr1"): P,
    ("B", "cr2"): P,
    ("C", "cr1"): F,
    ("C", "cr2"): P,
}
terra_script = {
    ("A", "cr1"): P,
    ("A", "cr2"): P,
    ("B", "cr1"): F,
    ("B", "cr2"): P,
    ("C", "cr1"): F,
    ("C", "cr2"): None,
}
sol_script = {
    ("A", "cr1"): P,
    ("A", "cr2"): F,
    ("B", "cr1"): P,
    ("B", "cr2"): P,
    ("C", "cr1"): F,
    ("C", "cr2"): P,
}

# the fakes ignore query/summary/articles, so dummy inputs are enough
crit = {c: Criterion(id=c, kind="judged", rule_text="x") for c in ("cr1", "cr2")}
items = [JudgeInput(case, "q", "s", (), crit[cr]) for case, cr in items_keys]

terra = FakeJudge(terra_script, config_version="fake-terra")
sol = FakeJudge(sol_script, config_version="fake-sol")


async def judge_vs_label(model: str, verdicts: list[CaseVerdict]) -> None:
    """Print one fake judge's calibration rows against ground_truth, with Wilson intervals."""
    for cal in await calibrate(verdicts, ground_truth):
        print(f"Fake {model} Scores : {cal}")
        w_range = await wilson(cal.agree, cal.total_items, 1.96)
        print(f"Fake {model} Wilson Range: [{w_range[0]:.4f}, {w_range[1]:.4f}]")


async def judge_vs_judge(terra_v: list[CaseVerdict], sol_v: list[CaseVerdict]) -> None:
    """Print the fake terra vs sol agreement rows, with Wilson intervals."""
    for agreement in await calibrate_2judges(("Terra", terra_v), ("Sol", sol_v)):
        print(f"Fake Judge 1 vs Judge2 Scores : {agreement}")
        w_range = await wilson(agreement.agree, agreement.total_items, 1.96)
        print(f"Fake terra vs sol Wilson Range: [{w_range[0]:.4f}, {w_range[1]:.4f}]")


async def main() -> None:
    """Run the judge-vs-judge and both judge-vs-label comparisons on the scripted verdicts."""
    terra_v, sol_v = terra(items), sol(items)
    await judge_vs_judge(
        terra_v, sol_v
    )  # expect: total=5, agree=3, only_j1=1, only_j2=1, errors_j1=1
    await judge_vs_label(
        "Terra", terra_v
    )  # expect: total=5, agree=3, false_pass=1, false_fail=1
    await judge_vs_label(
        "Sol", sol_v
    )  # expect: total=6, agree=6, false_pass=0, false_fail=0


if __name__ == "__main__":
    asyncio.run(main())
