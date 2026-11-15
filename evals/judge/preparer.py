"""Build judge inputs from the frozen e2e captures, the rubric and the human labels.

Pipeline (judge_agreement_runner.start_agreement_run):
  seed cases → judged_cases → load_e2e_fixture_captures → to_judge_input
  → input_for_judge_agreement (labelled items only) → NewsJudge
"""

import logging
from typing import Any

from evals.eval_config.load_eval_config import get_config
from evals.loader.loader import json_loader, yaml_loader
from evals.model.article_hit import ArticleHit
from evals.model.capture import Capture
from evals.model.judge_input import JudgeInput
from evals.model.models import Criterion

log = logging.getLogger(__name__)

GroundTruth = dict[tuple[str, str], bool]
"""Human labels: (case_id, criterion_id) → True for pass, False for fail."""


def _fixture_path(config_key: str, case_id: str, run_id: str) -> str:
    """A capture file path from its eval_config.yaml template (case_id / run_id placeholders)."""
    return (
        str(get_config(config_key))
        .replace("case_id", case_id)
        .replace("run_id", run_id)
    )


async def judged_cases(seed_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The seed cases the judge grades: every case whose `judge` is not "skip"."""
    return [case for case in seed_cases if case["judge"] != "skip"]


async def load_e2e_fixture_captures(
    run_id: str, seed_cases: list[dict[str, Any]]
) -> dict[str, Capture]:
    """Read response.json + state.json of each given seed case for one capture run.

    A case whose files are missing (e.g. a run made with --only, or a case added after
    the run) or hold JSON null is left out with a warning, so one gap does not stop the run.
    Returns case_id → Capture.
    """
    result: dict[str, Capture] = {}

    for case_id in await get_seed_case_ids(seed_cases=seed_cases):
        try:
            state = await json_loader(
                _fixture_path("CAPTURED_E2E_FIXTURE_STATE_PATH", case_id, run_id)
            )
            response = await json_loader(
                _fixture_path("CAPTURED_E2E_FIXTURE_RESPONSE_PATH", case_id, run_id)
            )
        except FileNotFoundError as exc:
            log.warning(
                "no capture for %s in run %s: %s", case_id, run_id, exc.filename
            )
            continue

        if state is None or response is None:
            log.warning("empty capture for %s in run %s", case_id, run_id)
            continue
        result.setdefault(
            case_id,
            Capture(
                case_id=case_id,
                run_id=run_id,
                http_status=response["http_status"],
                query=state["query"],
                summary=state["summary"],
                article_hits=await article_hits(state),
                response=response,
            ),
        )
    return result


async def article_hits(state: dict[str, Any]) -> tuple[ArticleHit, ...]:
    """The captured state's article_hits (mget output, content included) as ArticleHit tuples."""
    return tuple(
        ArticleHit(
            article_id=ah["article_id"],
            title=ah["title"],
            source=ah["source"],
            published_at=ah["published_at"],
            link=ah["link"],
            content=ah["content"],
        )
        for ah in state["article_hits"]
    )


async def get_seed_case_ids(seed_cases: list[dict[str, Any]]) -> list[str]:
    """The `id` of every seed case, in file order."""
    return [case["id"] for case in seed_cases]


async def get_case_labeled_ids(labeled_cases: dict[str, Any]) -> list[str]:
    """The `case_id` of every entry in a labels file (its `labels` list)."""
    return [case["case_id"] for case in labeled_cases["labels"]]


async def get_rubric() -> tuple[Criterion, ...]:
    """The judged criteria from rubric.yaml (`judge_criteria`), in file order."""
    rubric = await yaml_loader(get_config("SEED_RUBRIC_PATH"))
    return tuple(
        Criterion(id=key, kind=value["kind"], rule_text=value["rule_text"])
        for key, value in rubric["judge_criteria"].items()
    )


async def get_ground_truth(cases_path: str) -> GroundTruth:
    """The human labels from a labels file: (case_id, criterion_id) → passed.

    Every key of a label entry other than case_id, query and note is a criterion id
    whose value must be "pass" or "fail".
    """
    p_f_map = {"pass": True, "fail": False}
    cases: dict[str, Any] = await yaml_loader(relative_path=cases_path)
    return {
        (case["case_id"], attr): p_f_map[case[attr]]
        for case in cases["labels"]
        for attr in case
        if attr not in ["note", "case_id", "query"]
    }


async def to_judge_input(
    seed_cases: list[dict[str, Any]],
    captures: dict[str, Capture],
    rubric: tuple[Criterion, ...],
) -> list[JudgeInput]:
    """One JudgeInput per judged case with a gradable capture × each criterion the case lists.

    Left out: cases with `judge: skip`, cases without a capture, criteria not in the case's
    `judge.criteria`, and captures with no articles. A capture without articles is a fixed
    reply (off-topic, no results, search unavailable, summary failed), not a summary,
    so there is nothing to grade.
    """
    items: list[JudgeInput] = []
    for case in await judged_cases(seed_cases):
        capture = captures.get(case["id"])
        if capture is None:
            continue
        if not capture.article_hits:
            log.info(
                "%s: capture has no articles (fixed reply); not judged", case["id"]
            )
            continue
        wanted = set(case["judge"]["criteria"])
        items.extend(
            JudgeInput(
                case_id=case["id"],
                query=case["query"],
                summary=capture.summary,
                articles=capture.article_hits,
                criterion=cri,
            )
            for cri in rubric
            if cri.id in wanted
        )
    return items


async def input_for_judge_agreement(
    items: list[JudgeInput], ground_truth: GroundTruth
) -> list[JudgeInput]:
    """Only the items that have a human label (used to select / calibrate the judge)."""
    return [
        item
        for item in items
        if ground_truth.get((item.case_id, item.criterion.id)) is not None
    ]
