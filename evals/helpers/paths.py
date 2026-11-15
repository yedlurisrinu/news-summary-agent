"""Output paths for judge runs, built from the templates in eval_config.yaml.

Both are relative to the evals folder (BASE_PATH); loader and dumper join them.
"""

from evals.eval_config.load_eval_config import get_config


def judge_run_dir(captured_run_id: str, model_name: str, judge_version: str) -> str:
    """agreement_result/<captured_run_id>/<model_name>/<judge_version>"""
    return (
        str(get_config("JUDGE_VERDICTS_PATH"))
        .replace("captured-run-id", captured_run_id)
        .replace("model-name", model_name)
        .replace("judge-hash", judge_version)
    )


def judges_agreement_dir(captured_run_id: str) -> str:
    """agreement_result/<captured_run_id>"""
    return str(get_config("JUDGES_CALIBRATION_AGREEMENT_PATH")).replace(
        "captured-run-id", captured_run_id
    )
