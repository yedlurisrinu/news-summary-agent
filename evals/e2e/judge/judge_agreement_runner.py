"""Judge runs over the frozen e2e captures: grade (agreement run), judge vs labels, judge vs judge."""

import logging
from typing import cast

from langchain_openai import ChatOpenAI
from openai.types.shared import ReasoningEffort

from evals.e2e.dumper.dumper import dump_list_of_records
from evals.eval_config.load_eval_config import get_config
from evals.e2e.helpers.paths import judge_run_dir, judges_agreement_dir
from evals.e2e.helpers.utils import get_model
from evals.e2e.judge.calibration import calibrate, calibrate_2judges, wilson
from evals.e2e.judge.news_judge import NewsJudge
from evals.e2e.judge.preparer import (
    GroundTruth,
    get_ground_truth,
    input_for_judge_agreement,
    judged_cases,
    load_e2e_fixture_captures,
    to_judge_input,
)
from evals.e2e.loader.loader import json_loader, yaml_loader
from evals.e2e.model.judge_config import JudgeConfig
from evals.e2e.model.judge_input import JudgeInput
from evals.e2e.model.models import CaseVerdict

log = logging.getLogger(__name__)


async def _load_verdicts(
    captured_run_id: str, judge_config: JudgeConfig
) -> list[CaseVerdict]:
    """Read a judge's saved verdicts.json for one capture run."""
    run_dir = judge_run_dir(
        captured_run_id, judge_config.model_name, judge_config.version
    )
    return [
        CaseVerdict(**record)
        for record in await json_loader(f"{run_dir}/verdicts.json")
    ]


async def start_agreement_run(
    judge_config: JudgeConfig,
    captured_run_id: str,
    ground_truth: GroundTruth,
    temp: float = 0,
    total_runs: int = 5,
) -> None:
    """Grade every labelled (case, criterion) item of a capture run with one judge.

    Builds the judge inputs from the judged seed cases' captures, keeps only the items
    that have a human label, runs the judge total_runs times and saves the verdicts of
    the last run (by design) to <judge_run_dir>/verdicts.json. temp is passed to
    get_model, which currently ignores it (the gpt-6.1 judges only accept the default).
    """
    chat_model: ChatOpenAI = await get_model(
        model=judge_config.model_name,
        temperature=temp,
        timeout=judge_config.timeout_s,
        max_retries=judge_config.max_retries,
        reasoning_effort=cast(ReasoningEffort, judge_config.reasoning_effort),
    )

    seed_cases = (await yaml_loader(get_config("SEED_E2E_CASES_PATH")))["cases"]
    captures = await load_e2e_fixture_captures(
        run_id=captured_run_id, seed_cases=await judged_cases(seed_cases)
    )

    items: list[JudgeInput] = await input_for_judge_agreement(
        await to_judge_input(
            seed_cases=seed_cases, captures=captures, rubric=judge_config.rubric
        ),
        ground_truth,
    )

    # all the settings will come from the yaml file in integration; defaulted here
    # ignoring seed: not needed for the learning exercise
    news_judge = NewsJudge(
        judge_config, chat_model=chat_model, captured_run_id=captured_run_id
    )
    verdicts: list[CaseVerdict] = []
    for run in range(total_runs):
        verdicts = news_judge(items, run=run + 1)

    run_dir = judge_run_dir(
        captured_run_id, judge_config.model_name, judge_config.version
    )
    dump_list_of_records(verdicts, f"{run_dir}/verdicts.json")


async def judge_agreement(
    agreement_run: bool,
    judge_vs_label: bool,
    per_judge_agreement_runs: int,
    judge_1_config: JudgeConfig,
    judge_2_config: JudgeConfig | None,
) -> None:
    """One step of the judge workflow.

    agreement_run   → grade the labelled items with judge 1 and save verdicts.json
    judge_vs_label  → compare judge 1's saved verdicts with the human labels
    neither         → compare judge 1's and judge 2's saved verdicts (judge_2_config required)
    """
    ground_truth = await get_ground_truth(get_config("LABELED_E2E_CASES_PATH"))
    captured_run_id = str(get_config("CAPTURED_RUN_ID"))
    if agreement_run:
        await start_agreement_run(
            judge_1_config,
            captured_run_id=captured_run_id,
            total_runs=per_judge_agreement_runs,
            ground_truth=ground_truth,
        )
        return

    judge1_verdicts = await _load_verdicts(captured_run_id, judge_1_config)

    if judge_vs_label:
        calibrations = await calibrate(judge1_verdicts, ground_truth)
        run_dir = judge_run_dir(
            captured_run_id, judge_1_config.model_name, judge_1_config.version
        )
        dump_list_of_records(calibrations, f"{run_dir}/calibrations.json")
        for cal in calibrations:
            log.info(
                "%s vs labels [%s]: agree %d/%d, Wilson 95%% %s, false_pass %d, false_fail %d",
                judge_1_config.model_name,
                cal.criterion_id,
                cal.agree,
                cal.total_items,
                await wilson(cal.agree, cal.total_items, 1.96),
                cal.false_pass,
                cal.false_fail,
            )
        return

    if judge_2_config is None:
        raise ValueError("judge vs judge needs judge_2_config")
    judge2_verdicts = await _load_verdicts(captured_run_id, judge_2_config)
    judge_agreements = await calibrate_2judges(
        (judge_1_config.model_name, judge1_verdicts),
        (judge_2_config.model_name, judge2_verdicts),
    )
    dump_list_of_records(
        judge_agreements,
        f"{judges_agreement_dir(captured_run_id)}/judges_agreement-{judge_1_config.version}-{judge_2_config.version}.json",
    )
    for a in judge_agreements:
        log.info(
            "%s vs %s [%s]: agree %d/%d, Wilson 95%% %s, kappa %s",
            judge_1_config.model_name,
            judge_2_config.model_name,
            a.criterion_id,
            a.agree,
            a.total_items,
            await wilson(a.agree, a.total_items, 1.96) if a.total_items else None,
            a.kappa,
        )
