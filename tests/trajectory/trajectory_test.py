"""
tests/trajectory/trajectory_test.py — the stubbed trajectory suite (T05-S282).

One pytest test per case in evals/cases/trajectory_cases.yaml, plus the budget case
(TRJ-020, over the cases it lists) and the isolation case (TRJ-021, two runs on one
graph). Deterministic: scripted LLM + in-memory fake es-mcp-server. Runs in the
normal unit job, so a regression here blocks a merge.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from data.news_input import NewsInput
from evals.trajectory.harness import (
    ScriptedLLM,
    Trajectory,
    check,
    load_cases,
    mcp_from_case,
    outcome_of,
    run_trajectory,
)
from graph.build_graph import build
from graph.run_graph_workflow import run_graph

pytestmark = pytest.mark.anyio

PATHS, CASES = load_cases()
BY_ID = {c["id"]: c for c in CASES}
SIMPLE = [c for c in CASES if "expect" in c and "query" in c and "applies_to" not in c]


@pytest.mark.parametrize("case", SIMPLE, ids=[c["id"] for c in SIMPLE])
async def test_trajectory(case: dict[str, Any]) -> None:
    traj = await run_trajectory(
        case["query"], case.get("llm_script", []), case.get("mcp")
    )
    errors = check(case["expect"], traj, PATHS)
    assert not errors, f"{case['id']} {case['title']}:\n  " + "\n  ".join(errors)


BUDGET = BY_ID["TRJ-020"]


BUDGET_KEYS = {"each_node_at_most_once", "max_steps", "max_tool_calls", "max_llm_calls"}


def test_budget_case_has_only_known_keys() -> None:
    unknown = set(BUDGET["expect"]) - BUDGET_KEYS
    assert not unknown, f"TRJ-020 expect: unknown keys {unknown}"


@pytest.mark.parametrize("target", BUDGET["applies_to"])
async def test_budget(target: str) -> None:
    case, limits = BY_ID[target], BUDGET["expect"]
    traj = await run_trajectory(
        case["query"], case.get("llm_script", []), case.get("mcp")
    )
    if limits.get("each_node_at_most_once"):
        repeated = [n for n, k in Counter(traj.nodes).items() if k > 1]
        assert not repeated, f"{target}: nodes ran more than once: {repeated}"
    assert len(traj.nodes) <= limits["max_steps"], (
        f"{target}: {len(traj.nodes)} steps {traj.nodes}"
    )
    assert len(traj.tool_calls) <= limits["max_tool_calls"], (
        f"{target}: {traj.tool_calls}"
    )
    assert len(traj.llm_calls) <= limits["max_llm_calls"], f"{target}: {traj.llm_calls}"


async def test_back_to_back_runs_do_not_share_state() -> None:
    """TRJ-021: two queries on one graph and one checkpointer; run 2 sees only its own articles.

    Driven by the case's YAML: `expect.run_2` is checked with check() against run 2's
    tool calls and outcome; `expect.distinct_thread_ids` asserts one thread per run.
    """
    case = BY_ID["TRJ-021"]
    expect = case["expect"]
    unknown = set(expect) - {"run_2", "distinct_thread_ids"}
    assert not unknown, f"TRJ-021 expect: unknown keys {unknown}"
    first, second = case["sequence"]
    llm = ScriptedLLM(first["llm_script"] + second["llm_script"])
    fake = mcp_from_case(first["mcp"])
    saver = InMemorySaver()
    graph = await build(saver, llm.runnable, fake.client())  # type: ignore[arg-type]

    await run_graph(NewsInput(query=first["query"]), graph)
    second_fake = mcp_from_case(second["mcp"])
    fake.refs, fake.docs = second_fake.refs, second_fake.docs
    calls_before = len(fake.calls)
    response = await run_graph(NewsInput(query=second["query"]), graph)

    run_2 = Trajectory(
        tool_calls=list(fake.calls[calls_before:]), outcome=outcome_of(response)
    )
    errors = check(expect["run_2"], run_2, PATHS)
    assert not errors, "TRJ-021 run 2:\n  " + "\n  ".join(errors)
    if expect.get("distinct_thread_ids"):
        threads = {
            t.config["configurable"]["thread_id"] async for t in saver.alist(None)
        }
        assert len(threads) == 2, f"expected 2 distinct thread ids, got {len(threads)}"
