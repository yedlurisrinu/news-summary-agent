"""The LLM judge: grades (case, criterion) items with structured output (CriterionVerdict)."""

import json
from dataclasses import asdict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import OpenAIRefusalError
from openai import APIError, APITimeoutError, RateLimitError
from pydantic import ValidationError

from evals.e2e.dumper.dumper import dump_judge_config
from evals.e2e.helpers.paths import judge_run_dir
from evals.e2e.model.judge_config import JudgeConfig
from evals.e2e.model.judge_input import JudgeInput
from evals.e2e.model.models import CaseVerdict, CriterionVerdict
from evals.e2e.model.prompts import ARTICLE_SEPARATOR


class NewsJudge:
    """
    LLM judge for one JudgeConfig. Creating it writes judge_config.json next to the
    verdicts; calling it grades a list of JudgeInputs (one criterion per item).
    """

    def __init__(
        self, judge_config: JudgeConfig, chat_model: ChatOpenAI, captured_run_id: str
    ) -> None:
        """Write judge_config.json under the judge's run folder and build the prompt."""
        self.calls = 0
        # ignoring seed: not needed for the learning exercise
        dump_judge_config(
            judge_config=judge_config,
            path=f"{judge_run_dir(captured_run_id, judge_config.model_name, judge_config.version)}/judge_config.json",
        )
        self.config_version = judge_config.version
        self._chat_model = chat_model
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", judge_config.system_prompt),
                ("human", judge_config.human_prompt),
            ]
        )

    def __call__(self, use_cases: list[JudgeInput], run: int = 0) -> list[CaseVerdict]:
        """Grade every item once (up to 4 concurrent calls); one CaseVerdict per item, in input order.

        A failed call becomes a verdict with passed=None and error set to refusal, schema,
        timeout, rate_limit, api, or "<ExceptionType>: <message>" for anything unexpected.
        """
        self.calls += 1
        verdicts: list[CaseVerdict] = []

        chain = self._prompt | self._chat_model.with_structured_output(
            CriterionVerdict
        )  # CriterionVerdict = evidence, passed only
        results = chain.batch(
            [
                {
                    "articles": ARTICLE_SEPARATOR.join(
                        [
                            json.dumps(asdict(article), ensure_ascii=False)
                            for article in use_case.articles
                        ]
                    ),
                    "summary": use_case.summary,
                    "query": use_case.query,
                    "criterion": use_case.criterion.rule_text,
                }
                for use_case in use_cases
            ],
            return_exceptions=True,
            config={"max_concurrency": 4},
        )
        # item is the JudgeInput sent; r is its verdict or exception (batch keeps input order)
        for item, r in zip(use_cases, results, strict=True):
            if isinstance(r, CriterionVerdict):
                verdicts.append(
                    CaseVerdict(
                        evidence=r.evidence,
                        passed=r.passed,
                        case_id=item.case_id,
                        criterion_id=item.criterion.id,
                        judge_config_version=self.config_version,
                        run=run,
                        error=None,
                    )
                )
                continue

            if isinstance(r, OpenAIRefusalError):
                error = "refusal"
            elif isinstance(r, ValidationError):
                error = "schema"
            elif isinstance(r, APITimeoutError):
                error = "timeout"  # before APIError: it's a subclass
            elif isinstance(r, RateLimitError):
                error = "rate_limit"
            elif isinstance(r, APIError):
                error = "api"
            else:  # an unexpected bug, not a judge error
                error = f"{type(r).__name__}: {r}"
            verdicts.append(
                CaseVerdict(
                    evidence=None,
                    passed=None,
                    case_id=item.case_id,
                    criterion_id=item.criterion.id,
                    judge_config_version=self.config_version,
                    run=run,
                    error=error,
                )
            )
        return verdicts
