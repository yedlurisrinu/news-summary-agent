"""Agreement maths: judge vs human labels, judge vs judge, Cohen's kappa and the Wilson interval.

Both comparisons return one row for all criteria pooled (criterion_id "all") followed by
one row per criterion (sorted by id). Read the per-criterion rows first: pooling mixes
criteria of different difficulty, and the pooled items are not independent (each case
contributes one item per criterion), so a Wilson interval on the pooled row is too narrow.
"""

import logging
from collections.abc import Iterable
from math import sqrt

from evals.model.judge_agreement import JudgeAgreement
from evals.model.models import Calibration, CaseVerdict

log = logging.getLogger(__name__)

ALL_CRITERIA = "all"


def _criterion_groups(criterion_ids: Iterable[str]) -> list[str]:
    """ "all" first, then each criterion id once, sorted."""
    return [ALL_CRITERIA, *sorted(set(criterion_ids))]


def _in_group(criterion_id: str, group: str) -> bool:
    """True when a verdict for criterion_id belongs to the group ("all" holds every criterion)."""
    return group == ALL_CRITERIA or criterion_id == group


def _calibrate_group(
    group: str,
    verdicts: list[CaseVerdict],
    ground_truth: dict[tuple[str, str], bool],
) -> Calibration | None:
    """Judge vs labels for one group of verdicts; None when no verdict was answered."""
    errored = agree = hp = ff = hf = fp = 0
    for verdict in verdicts:
        if verdict.passed is None:
            errored += 1
            continue
        human_feedback = ground_truth[verdict.case_id, verdict.criterion_id]
        agree += human_feedback == verdict.passed
        if human_feedback:
            hp += 1  # human passes
            ff += not verdict.passed  # judge false fails
        else:
            hf += 1  # human fails
            fp += verdict.passed  # judge false passes
    if hp + hf == 0:
        return None
    return Calibration(
        criterion_id=group,
        total_items=len(verdicts) - errored,
        agree=agree,
        false_pass=fp,
        false_fail=ff,
        human_fails=hf,
        human_passes=hp,
    )


async def calibrate(
    verdicts: list[CaseVerdict], ground_truth: dict[tuple[str, str], bool]
) -> list[Calibration]:
    """Compare one judge's verdicts with the human labels; errored verdicts are left out.

    Returns the pooled row ("all") and one row per criterion; a group in which every
    verdict errored has no row. Every verdict's (case_id, criterion_id) must be labelled.
    """
    for verdict in verdicts:
        if verdict.passed is None:
            log.warning(
                "judge failed with error: %s for case id %s, criterion id %s in run %s",
                verdict.error,
                verdict.case_id,
                verdict.criterion_id,
                verdict.run,
            )
    calibrations = [
        _calibrate_group(
            group,
            [v for v in verdicts if _in_group(v.criterion_id, group)],
            ground_truth,
        )
        for group in _criterion_groups(v.criterion_id for v in verdicts)
    ]
    return [c for c in calibrations if c is not None]


def cohen_kappa(
    both_pass: int, only_j1: int, only_j2: int, both_fail: int
) -> float | None:
    """Cohen's kappa for two raters on pass/fail items, from the 2x2 agreement table.

    kappa = (p_o - p_e) / (1 - p_e): observed agreement p_o corrected for the agreement p_e
    expected by chance from each rater's own pass rate. 1 = perfect, 0 = chance level,
    < 0 = worse than chance. None when undefined: no items, or p_e = 1 (both raters gave
    the same single verdict to every item, so there is no variation to agree on).
    """
    n = both_pass + only_j1 + only_j2 + both_fail
    if n == 0:
        return None
    p1 = (both_pass + only_j1) / n  # judge 1 pass rate
    p2 = (both_pass + only_j2) / n  # judge 2 pass rate
    p_e = p1 * p2 + (1 - p1) * (1 - p2)
    if p_e == 1:
        return None
    p_o = (both_pass + both_fail) / n
    return round((p_o - p_e) / (1 - p_e), 4)


def _agreement_group(
    group: str,
    j1_name: str,
    j2_name: str,
    pairs: list[tuple[CaseVerdict, CaseVerdict]],
) -> JudgeAgreement:
    """Judge 1 vs judge 2 for one group of (j1, j2) verdict pairs."""
    skipped = j1_error = j2_error = 0  # skipped: items at least one judge errored on
    both_pass = both_fail = j1_only = j2_only = 0
    for j1_verdict, j2_verdict in pairs:
        if j1_verdict.passed is None or j2_verdict.passed is None:
            skipped += 1
            j1_error += j1_verdict.error is not None
            j2_error += j2_verdict.error is not None
            continue
        if j1_verdict.passed and j2_verdict.passed:
            both_pass += 1
        elif j1_verdict.passed:
            j1_only += 1
        elif j2_verdict.passed:
            j2_only += 1
        else:
            both_fail += 1

    return JudgeAgreement(
        criterion_id=group,
        judge_1=j1_name,
        judge_2=j2_name,
        # count skipped items, not errors: an item both judges errored on is one item
        total_items=len(pairs) - skipped,
        agree=both_pass + both_fail,
        only_j1=j1_only,
        only_j2=j2_only,
        errors_j1=j1_error,
        errors_j2=j2_error,
        kappa=cohen_kappa(both_pass, j1_only, j2_only, both_fail),
    )


async def calibrate_2judges(
    j1_verdicts: tuple[str, list[CaseVerdict]],
    j2_verdicts: tuple[str, list[CaseVerdict]],
) -> list[JudgeAgreement]:
    """Compare two judges item by item; an item either judge errored on is left out.

    Each tuple is (judge name, verdicts). The two lists are paired by position, so they
    must hold the same items in the same order: a length mismatch (zip strict=True) or a
    pair whose (case_id, criterion_id) differ raises ValueError. The judge hash does not
    cover the cases, labels or captures, so two runs saved apart can drift out of step.

    Returns the pooled row ("all") and one row per criterion.
    """
    j1_name, j1_list = j1_verdicts
    j2_name, j2_list = j2_verdicts
    pairs = list(zip(j1_list, j2_list, strict=True))
    for pos, (j1_verdict, j2_verdict) in enumerate(pairs):
        j1_key = (j1_verdict.case_id, j1_verdict.criterion_id)
        j2_key = (j2_verdict.case_id, j2_verdict.criterion_id)
        if j1_key != j2_key:
            raise ValueError(
                f"verdicts out of step at position {pos}: {j1_name} has {j1_key}, "
                f"{j2_name} has {j2_key}; both judges must grade the same items in the same order"
            )

    return [
        _agreement_group(
            group,
            j1_name,
            j2_name,
            [p for p in pairs if _in_group(p[0].criterion_id, group)],
        )
        for group in _criterion_groups(v.criterion_id for v in j1_list)
    ]


async def wilson(passed: int, total_items: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for the proportion passed / total_items.

    passed: number of agreeing items
    total_items: items sent to the judge; one item per case per criterion
    z: z-score of the target confidence level (1.96 → 95%)
    Assumes independent items: use it on per-criterion rows, not the pooled "all" row.
    Raises ValueError when total_items is 0.
    """
    if not total_items:
        raise ValueError(" total items can not be zero ")
    pass_rate = passed / total_items
    # (p̂ + z²/2n) / (1 + z²/n)
    center = (pass_rate + (z**2) / (2 * total_items)) / (1 + (z**2) / total_items)
    # z · √(p̂(1−p̂)/n + z²/4n²) / (1 + z²/n)
    half_width = (
        z
        * sqrt(pass_rate * (1 - pass_rate) / total_items + z**2 / (4 * total_items**2))
    ) / (1 + z**2 / total_items)

    return (round(center - half_width, 4), round(center + half_width, 4))
