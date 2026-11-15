"""Judge calibration / agreement driver.

  uv run python -m evals.main

Runs each entry of JUDGE_AGREEMENT_RUNS_PARAMS (evals/resources/eval_config.yaml) in
order and asks before the next one ("q" stops). Needs Vault secrets (OpenAI key).
"""

import asyncio
import logging

from config.application_config import load_config as app_load_config
from config.logging_setup import setup_logging
from config.vault_secrets import load_secrets
from evals.eval_config.load_eval_config import get_config
from evals.eval_config.load_eval_config import load_config as eval_load_config
from evals.judge.judge_agreement_runner import judge_agreement
from evals.judge.preparer import get_rubric
from evals.model.judge_config import JudgeConfig
from evals.model.models import Criterion, CriterionVerdict
from evals.model.prompts import ARTICLE_SEPARATOR, HUMAN_PROMPT, SYSTEM_INSTRUCTIONS

log = logging.getLogger(__name__)


def get_judge_config(judge: str, rubric: tuple[Criterion, ...]) -> JudgeConfig:
    """The JudgeConfig for one judge model; its version names the agreement_result folder.

    Every field is part of JudgeConfig.version: changing any of them needs a new agreement run.
    """
    return JudgeConfig(
        model_name=judge,
        reasoning_effort="medium",
        timeout_s=int(get_config("LLM_TIMEOUT")),
        max_retries=int(get_config("LLM_MAX_RETRIES")),
        system_prompt=SYSTEM_INSTRUCTIONS,
        human_prompt=HUMAN_PROMPT,
        article_separator=ARTICLE_SEPARATOR,
        rubric=rubric,
        verdict_schema=CriterionVerdict.model_json_schema(),
    )


async def main() -> None:
    """Run each JUDGE_AGREEMENT_RUNS_PARAMS row in order, asking before the next one."""
    judge_agreement_run_params = get_config("JUDGE_AGREEMENT_RUNS_PARAMS")

    rubric = await get_rubric()

    # params: [judge_1, judge_2, agreement_run, judge_vs_label, per_judge_runs, label]
    for params in judge_agreement_run_params:
        judge_1_config = get_judge_config(params[0], rubric)
        judge_2_config = None
        if not params[2] and not params[3]:
            judge_2_config = get_judge_config(params[1], rubric)

        await judge_agreement(
            agreement_run=params[2],
            judge_vs_label=params[3],
            per_judge_agreement_runs=params[4],
            judge_1_config=judge_1_config,
            judge_2_config=judge_2_config,
        )
        done = (
            f"{params[5]} completed for input judge1={judge_1_config.model_name}, "
            f"judge2={judge_2_config.model_name if judge_2_config is not None else None}, "
            f"agreement_run={params[2]}, judge_vs_label={params[3]}, "
            f"per_judge_agreement_runs={params[4]}"
        )
        log.info(done)
        if input(f"{done}\nShould we continue next run ? ('q' to stop) ") == "q":
            return


if __name__ == "__main__":
    setup_logging()
    load_secrets()
    app_load_config()
    eval_load_config()
    asyncio.run(main())
