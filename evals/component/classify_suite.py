"""Component suite — classify (component_cases.yaml → classify, pass_criteria.classify).

  uv run python -m evals.component.classify_suite            # real app LLM (needs Vault secrets)
  uv run python -m evals.component.classify_suite --llm fake # pipeline check, no API calls
  uv run python -m evals.component.classify_suite --llm real # for real API call
Grades the app's own classification prompt (nodes.classification._CATEGORY_PROMPT) with the
app's own chat model, exactly as the classify node calls it, without the MCP category lookup:
the component under test is the LLM's JSON output. Code-graded only (no judge).

Writes evals/<COMPONENT_RESULTS_PATH>/classify-<run_id>.json: provenance, per-case outputs and
checks, and every suite gate with counts, Wilson interval and verdict (met / missed / inconclusive).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import combinations
from typing import Any

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableLambda

from evals.eval_config.load_eval_config import get_config
from evals.e2e.helpers.constants import EVAL_BASE_PATH
from evals.e2e.judge.calibration import wilson
from nodes.classification import _CATEGORY_PROMPT

log = logging.getLogger(__name__)

_ENGLISH = re.compile(r"^[a-z][a-z ]*$")


# --------------------------------------------------------------------------- data
@dataclass(frozen=True)
class Taxonomy:
    allowed: frozenset[str]
    alias_to_canonical: dict[str, str]

    def canon(self, category: str) -> str:
        c = category.strip().lower()
        return self.alias_to_canonical.get(c, c)

    def canon_set(self, categories: list[str]) -> set[str]:
        return {self.canon(c) for c in categories}


@dataclass
class Output:
    raw: str
    parsed: dict[str, Any] | None
    json_valid: bool
    error: str | None = None


@dataclass
class CaseResult:
    case_id: str
    query: str
    outputs: list[Output]
    checks: dict[str, bool | None] = field(default_factory=dict)  # None = not scored

    @property
    def passed(self) -> bool:
        return all(v for v in self.checks.values() if v is not None)


def load_classify(
    path: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], Taxonomy]:
    file = EVAL_BASE_PATH / (path or get_config("SEED_COMPONENT_CASES_PATH"))
    data = yaml.safe_load(file.read_text(encoding="utf-8"))
    shared = data["shared"]
    alias_to_canonical = {
        alias: canonical
        for canonical, aliases in shared.get("category_aliases", {}).items()
        for alias in aliases
    }
    allowed = frozenset(
        alias_to_canonical.get(c, c) for c in shared["allowed_categories"]
    )
    return (
        data["classify"],
        data["pass_criteria"]["classify"],
        Taxonomy(allowed, alias_to_canonical),
    )


# --------------------------------------------------------------------------- one call
def _valid_shape(parsed: Any) -> bool:
    return (
        isinstance(parsed, dict)
        and isinstance(parsed.get("is_news_query"), bool)
        and isinstance(parsed.get("categories"), list)
        and all(isinstance(c, str) for c in parsed["categories"])
        and isinstance(parsed.get("fetch_latest"), bool)
    )


async def classify_once(llm: Runnable[Any, Any], query: str) -> Output:
    """The classify node's own prompt + model; raw text kept so invalid JSON is visible."""
    try:
        message = await (_CATEGORY_PROMPT | llm).ainvoke({"query": query})
    except Exception as exc:  # noqa: BLE001 - a failed call is a recorded output, not a crash
        return Output(
            raw="", parsed=None, json_valid=False, error=f"{type(exc).__name__}: {exc}"
        )
    raw = str(getattr(message, "content", message))
    try:
        parsed = JsonOutputParser().parse(raw)
    except Exception as exc:  # noqa: BLE001
        return Output(raw=raw, parsed=None, json_valid=False, error=f"parse: {exc}")
    return Output(raw=raw, parsed=parsed, json_valid=_valid_shape(parsed))


# --------------------------------------------------------------------------- scoring (pure)
def jaccard(a: set[str], b: set[str]) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def score_case(
    case: dict[str, Any], outputs: list[Output], tax: Taxonomy
) -> CaseResult:
    query = case["input"]["query"]
    result = CaseResult(case["id"], query, outputs)

    if (
        "expected" not in case
    ):  # stability case: compare repeated outputs with each other
        valid = [o.parsed for o in outputs if o.json_valid and o.parsed is not None]
        all_valid = len(valid) == len(outputs)
        result.checks["json_valid"] = all_valid
        result.checks["is_news_query_stable"] = (
            all_valid and len({p["is_news_query"] for p in valid}) == 1
        )
        result.checks["fetch_latest_stable"] = (
            all_valid and len({p["fetch_latest"] for p in valid}) == 1
        )
        sets = [tax.canon_set(p["categories"]) for p in valid]
        min_j = min((jaccard(a, b) for a, b in combinations(sets, 2)), default=1.0)
        result.checks["categories_jaccard_gte_0.6"] = all_valid and min_j >= 0.6
        return result

    out = outputs[0]
    exp = case["expected"]
    result.checks["json_valid"] = out.json_valid
    if not out.json_valid or out.parsed is None:
        for name in (
            "is_news_query",
            "fetch_latest",
            "categories_must",
            "categories_in_bounds",
        ):
            result.checks[name] = False
        return result

    p = out.parsed
    got = tax.canon_set(p["categories"])
    must = tax.canon_set(exp.get("categories_must", []))
    may = tax.canon_set(exp.get("categories_may", []))
    if exp.get("is_news_query") is not None:
        result.checks["is_news_query"] = p["is_news_query"] == exp["is_news_query"]
    if exp.get("fetch_latest") is not None:
        result.checks["fetch_latest"] = p["fetch_latest"] == exp["fetch_latest"]
    result.checks["categories_must"] = must <= got
    result.checks["categories_in_bounds"] = got <= (must | may)
    for extra in case.get("deterministic", []):
        if extra == "output_is_valid_json":
            result.checks[extra] = out.json_valid
        elif extra == "categories_in_english":
            result.checks[extra] = all(
                _ENGLISH.match(c.strip().lower()) for c in p["categories"]
            )
        elif extra == "categories_subset_of_allowed":
            result.checks[extra] = got <= tax.allowed
    if (
        "judge" in case
    ):  # e.g. CMP-CLS-013 categories_reasonable: no classify judge in this gate
        for criterion in case["judge"].get("criteria", []):
            result.checks[f"judged:{criterion}"] = None
    return result


@dataclass
class Gate:
    name: str
    passed: int
    total: int
    bar: float
    wilson: tuple[float, float] | None = None
    verdict: str = ""


def _bar(text: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    return float(m.group(1)) / 100 if m else 1.0


async def evaluate_gate(name: str, passed: int, total: int, bar: float) -> Gate:
    """Bar 100%: met only if every item passes. Otherwise the three-way Wilson rule."""
    gate = Gate(name, passed, total, bar)
    if total == 0:
        gate.verdict = "no_data"
        return gate
    gate.wilson = await wilson(passed, total)
    lo, hi = gate.wilson
    if bar >= 1.0:
        gate.verdict = "met" if passed == total else "missed"
    elif lo >= bar:
        gate.verdict = "met"
    elif hi < bar:
        gate.verdict = "missed"
    else:
        gate.verdict = "inconclusive"
    return gate


async def suite_gates(
    cases: list[dict[str, Any]], results: list[CaseResult], criteria: dict[str, Any]
) -> list[Gate]:
    by_id = {c["id"]: c for c in cases}
    scored = [r for r in results if "expected" in by_id[r.case_id]]
    stability = [r for r in results if "expected" not in by_id[r.case_id]]
    sg = criteria["suite_gates"]

    def count(rs: list[CaseResult], check: str) -> tuple[int, int]:
        vals = [r.checks[check] for r in rs if r.checks.get(check) is not None]
        return sum(bool(v) for v in vals), len(vals)

    outputs = [o for r in results for o in r.outputs]
    exp = lambda r: by_id[r.case_id]["expected"]
    off = [r for r in scored if exp(r).get("is_news_query") is False]
    news = [r for r in scored if exp(r).get("is_news_query") is True]
    with_must = [r for r in scored if exp(r).get("categories_must")]

    gates = [
        await evaluate_gate(
            "json_valid_rate",
            sum(o.json_valid for o in outputs),
            len(outputs),
            _bar(sg["json_valid_rate"]),
        ),
        await evaluate_gate(
            "off_topic_recall",
            *count(off, "is_news_query"),
            _bar(sg["off_topic_recall"]),
        ),
        await evaluate_gate(
            "is_news_query_recall",
            *count(news, "is_news_query"),
            _bar(sg["is_news_query_recall"]),
        ),
        await evaluate_gate(
            "fetch_latest_accuracy",
            *count(scored, "fetch_latest"),
            _bar(sg["fetch_latest_accuracy"]),
        ),
        await evaluate_gate(
            "categories_must_hit_rate",
            *count(with_must, "categories_must"),
            _bar(sg["categories_must_hit_rate"]),
        ),
        # stated as out_of_bounds <= 10%: gate on the in-bounds share >= 90%
        await evaluate_gate(
            "categories_in_bounds_rate",
            *count(scored, "categories_in_bounds"),
            1 - _bar(sg["categories_out_of_bounds_rate"]),
        ),
    ]
    for r in stability:
        ok = all(v for v in r.checks.values() if v is not None)
        gates.append(await evaluate_gate(f"stability_{r.case_id}", int(ok), 1, 1.0))
    return gates


# --------------------------------------------------------------------------- run
def fake_llm(cases: list[dict[str, Any]]) -> Runnable[Any, Any]:
    """Answers each query with its expected values: the whole pipeline should report every gate met."""
    expected: dict[str, dict[str, Any]] = {}
    for c in cases:
        e = c.get("expected") or next(
            x["expected"]
            for x in cases
            if x["input"]["query"] == c["input"]["query"] and "expected" in x
        )
        cats = (
            list(e.get("categories_must") or e.get("categories_may", [])[:1])
            if e.get("is_news_query")
            else []
        )
        expected[c["input"]["query"]] = {
            "is_news_query": bool(e.get("is_news_query")),
            "categories": cats,
            "fetch_latest": bool(e.get("fetch_latest")),
        }

    def respond(prompt: PromptValue) -> AIMessage:
        query = str(prompt.to_messages()[-1].content)
        return AIMessage(content=json.dumps(expected[query]))

    return RunnableLambda(respond)


def real_llm() -> BaseChatModel:
    from adapters.llm import get_chat_model
    from config.application_config import get_config as app_config

    return get_chat_model(
        str(app_config("LLM_MODEL")),
        float(app_config("LLM_TEMPERATURE")),
        int(app_config("LLM_TIMEOUT_SEC")),
    )


async def run(llm_kind: str) -> dict[str, Any]:
    cases, criteria, tax = load_classify()
    llm: Runnable[Any, Any] = fake_llm(cases) if llm_kind == "fake" else real_llm()

    results: list[CaseResult] = []
    for case in cases:
        repeat = int(case["input"].get("repeat", 1))
        outputs = [
            await classify_once(llm, case["input"]["query"]) for _ in range(repeat)
        ]
        results.append(score_case(case, outputs, tax))
    gates = await suite_gates(cases, results, criteria)

    prompt_text = "\n".join(
        str(m.prompt.template)
        for m in _CATEGORY_PROMPT.messages
        if hasattr(m, "prompt")
    )
    cases_file = EVAL_BASE_PATH / get_config("SEED_COMPONENT_CASES_PATH")
    report = {
        "suite": "classify",
        "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "llm": llm_kind,
        "model": None if llm_kind == "fake" else getattr(llm, "model_name", None),
        "prompt_sha": hashlib.sha256(prompt_text.encode()).hexdigest()[:12],
        "cases_sha": hashlib.sha256(cases_file.read_bytes()).hexdigest()[:12],
        "cases_passed": sum(r.passed for r in results),
        "cases_total": len(results),
        "gates": [g.__dict__ for g in gates],
        "cases": [
            {
                "case_id": r.case_id,
                "query": r.query,
                "passed": r.passed,
                "checks": r.checks,
                "outputs": [o.__dict__ for o in r.outputs],
            }
            for r in results
        ],
    }
    out_dir = EVAL_BASE_PATH / get_config("COMPONENT_RESULTS_PATH")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"classify-{report['run_id']}-{llm_kind}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def print_summary(report: dict[str, Any]) -> None:
    print(
        f"classify ({report['llm']}): {report['cases_passed']}/{report['cases_total']} cases passed"
    )
    for g in report["gates"]:
        print(
            f"  {g['name']:<28} {g['passed']:>3}/{g['total']:<3} bar {g['bar']:.2f}  wilson {g['wilson']}  -> {g['verdict']}"
        )
    for c in report["cases"]:
        if not c["passed"]:
            failed = [k for k, v in c["checks"].items() if v is False]
            print(
                f"  FAIL {c['case_id']}: {failed}  output={c['outputs'][0]['raw'][:120]!r}"
            )


if __name__ == "__main__":
    from config.application_config import load_config as app_load_config
    from config.logging_setup import setup_logging
    from evals.eval_config.load_eval_config import load_config as eval_load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["fake", "real"], default="fake")
    args = parser.parse_args()
    setup_logging()
    app_load_config()
    eval_load_config()
    if args.llm == "real":
        from config.vault_secrets import load_secrets

        load_secrets()
    print_summary(asyncio.run(run(args.llm)))