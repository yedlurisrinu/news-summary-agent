"""
tests/evals/calibration_test.py

The judge agreement maths in evals/judge/calibration.py, with scripted judges (FakeJudge):
  - calibrate          — judge vs human labels; errored verdicts are left out
  - calibrate_2judges  — judge vs judge; an item either judge errored on counts once
  - per-criterion rows — 'all' first, then one row per criterion
  - cohen_kappa        — matches scikit-learn; None when undefined
  - wilson             — matches statsmodels' Wilson interval; rejects n = 0
Scenario: evals/helpers/fake_script.py (3 cases x 2 criteria).
"""

from __future__ import annotations

import pytest
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.proportion import proportion_confint

from evals.helpers.fake_judge import FakeJudge
from evals.judge.calibration import (
    calibrate,
    calibrate_2judges,
    cohen_kappa,
    wilson,
)
from evals.model.judge_input import JudgeInput
from evals.model.models import CaseVerdict, Criterion

pytestmark = pytest.mark.anyio

P, F = True, False
KEYS = [
    ("A", "cr1"),
    ("A", "cr2"),
    ("B", "cr1"),
    ("B", "cr2"),
    ("C", "cr1"),
    ("C", "cr2"),
]
GROUND_TRUTH = dict(zip(KEYS, [P, F, P, P, F, P], strict=True))
TERRA = dict(
    zip(KEYS, [P, P, F, P, F, None], strict=True)
)  # 1 false pass, 1 false fail, 1 error
SOL = dict(zip(KEYS, [P, F, P, P, F, P], strict=True))  # matches the labels exactly

_CRIT = {c: Criterion(id=c, kind="judged", rule_text="x") for c in ("cr1", "cr2")}
ITEMS = [JudgeInput(case, "q", "s", (), _CRIT[cr]) for case, cr in KEYS]


def _verdicts(script: dict[tuple[str, str], bool | None]) -> list[CaseVerdict]:
    return FakeJudge(script)(ITEMS, run=1)


def _rows(rows: list) -> dict:
    return {r.criterion_id: r for r in rows}


class TestCalibrate:
    async def test_pooled_row_first_then_one_per_criterion(self) -> None:
        rows = await calibrate(_verdicts(TERRA), GROUND_TRUTH)
        assert [r.criterion_id for r in rows] == ["all", "cr1", "cr2"]

    async def test_judge_with_errors_and_mistakes(self) -> None:
        rows = _rows(await calibrate(_verdicts(TERRA), GROUND_TRUTH))
        pooled, cr1, cr2 = rows["all"], rows["cr1"], rows["cr2"]
        assert pooled.total_items == 5  # the errored C/cr2 is left out
        assert (pooled.agree, pooled.false_pass, pooled.false_fail) == (3, 1, 1)
        assert (pooled.human_passes, pooled.human_fails) == (3, 2)
        # per criterion: cr1 = A, B, C; cr2 = A, B (C errored)
        assert (cr1.total_items, cr1.agree, cr1.false_pass, cr1.false_fail) == (
            3,
            2,
            0,
            1,
        )
        assert (cr2.total_items, cr2.agree, cr2.false_pass, cr2.false_fail) == (
            2,
            1,
            1,
            0,
        )

    async def test_perfect_judge(self) -> None:
        rows = _rows(await calibrate(_verdicts(SOL), GROUND_TRUTH))
        assert {k: (r.total_items, r.agree) for k, r in rows.items()} == {
            "all": (6, 6),
            "cr1": (3, 3),
            "cr2": (3, 3),
        }

    async def test_all_errored_gives_no_calibration(self) -> None:
        assert await calibrate(_verdicts(dict.fromkeys(KEYS)), GROUND_TRUTH) == []

    async def test_criterion_with_only_errors_has_no_row(self) -> None:
        cr2_errored = {k: (v if k[1] == "cr1" else None) for k, v in SOL.items()}
        rows = await calibrate(_verdicts(cr2_errored), GROUND_TRUTH)
        assert [r.criterion_id for r in rows] == ["all", "cr1"]


class TestCalibrate2Judges:
    async def test_terra_vs_sol(self) -> None:
        rows = _rows(
            await calibrate_2judges(
                ("Terra", _verdicts(TERRA)), ("Sol", _verdicts(SOL))
            )
        )
        assert list(rows) == ["all", "cr1", "cr2"]
        pooled = rows["all"]
        assert pooled.judge_1 == "Terra" and pooled.judge_2 == "Sol"
        assert pooled.total_items == 5
        assert (pooled.agree, pooled.only_j1, pooled.only_j2) == (3, 1, 1)
        assert (pooled.errors_j1, pooled.errors_j2) == (1, 0)
        assert (rows["cr1"].total_items, rows["cr1"].agree, rows["cr1"].only_j2) == (
            3,
            2,
            1,
        )
        assert (rows["cr2"].total_items, rows["cr2"].agree, rows["cr2"].only_j1) == (
            2,
            1,
            1,
        )
        assert rows["cr2"].errors_j1 == 1

    async def test_kappa_matches_sklearn(self) -> None:
        terra, sol = _verdicts(TERRA), _verdicts(SOL)
        (pooled, *_) = await calibrate_2judges(("Terra", terra), ("Sol", sol))
        answered = [
            (t.passed, s.passed)
            for t, s in zip(terra, sol, strict=True)
            if t.passed is not None and s.passed is not None
        ]
        expected = cohen_kappa_score([a for a, _ in answered], [b for _, b in answered])
        assert pooled.kappa == pytest.approx(expected, abs=1e-4)

    async def test_item_both_judges_errored_on_counts_once(self) -> None:
        both_err = {**SOL, ("C", "cr2"): None}
        (pooled, *_) = await calibrate_2judges(
            ("Terra", _verdicts(TERRA)), ("Sol", _verdicts(both_err))
        )
        assert pooled.total_items == 5  # 6 items, 1 skipped — not 6 - (1 + 1)
        assert (pooled.errors_j1, pooled.errors_j2) == (1, 1)

    async def test_same_length_different_order_raises(self) -> None:
        sol = _verdicts(SOL)
        sol[0], sol[1] = (
            sol[1],
            sol[0],
        )  # e.g. a labels file edited between the two runs
        with pytest.raises(ValueError, match="out of step at position 0"):
            await calibrate_2judges(("Terra", _verdicts(TERRA)), ("Sol", sol))

    async def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            await calibrate_2judges(
                ("Terra", _verdicts(TERRA)), ("Sol", _verdicts(SOL)[:-1])
            )


class TestCohenKappa:
    @pytest.mark.parametrize(
        "table", [(2, 1, 1, 1), (10, 0, 0, 5), (3, 2, 4, 1), (0, 5, 5, 0)]
    )
    def test_matches_sklearn(self, table: tuple[int, int, int, int]) -> None:
        both_pass, only_j1, only_j2, both_fail = table
        j1 = (
            [True] * both_pass
            + [True] * only_j1
            + [False] * only_j2
            + [False] * both_fail
        )
        j2 = (
            [True] * both_pass
            + [False] * only_j1
            + [True] * only_j2
            + [False] * both_fail
        )
        assert cohen_kappa(*table) == pytest.approx(cohen_kappa_score(j1, j2), abs=1e-4)

    def test_undefined_when_no_variation_or_no_items(self) -> None:
        assert cohen_kappa(7, 0, 0, 0) is None  # both judges passed everything
        assert cohen_kappa(0, 0, 0, 0) is None


class TestWilson:
    @pytest.mark.parametrize(
        ("passed", "n"), [(0, 10), (3, 5), (6, 6), (14, 16), (50, 100)]
    )
    async def test_matches_statsmodels(self, passed: int, n: int) -> None:
        lo, hi = await wilson(passed, n, 1.96)
        ref_lo, ref_hi = proportion_confint(passed, n, alpha=0.05, method="wilson")
        assert lo == pytest.approx(ref_lo, abs=1e-3)
        assert hi == pytest.approx(ref_hi, abs=1e-3)

    async def test_zero_items_raises(self) -> None:
        with pytest.raises(ValueError):
            await wilson(0, 0)
