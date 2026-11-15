"""
evals/trajectory/harness.py — run one stubbed trajectory case and check it.

A case (evals/cases/trajectory_cases.yaml) scripts the LLM (`llm_script`) and the MCP
server (`mcp`), runs the app's real graph, and asserts on what happened:
  nodes       node order, from graph.astream(stream_mode="updates")
  tool_calls  (tool, args) the fake MCP server received
  llm_calls   classify | summarize, in order
  outcome     HTTP status, status field, error / canned reply name
Fully deterministic: no API key, no Elasticsearch.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from data.news_input import NewsInput
from data.query_failed_response import QueryFailedResponse
from data.seed_data import SeedData
from exception.search_unavailable import SEARCH_UNAVAILABLE_MESSAGE
from graph.build_graph import build
from graph.run_graph_workflow import QUERY_FAILED_MESSAGE, run_graph
from nodes.summary import SUMMARY_FAILED_MESSAGE
from tests.fake_mcp import FakeNewsMcp, make_doc

CASES_FILE = Path(__file__).resolve().parent.parent / "cases" / "trajectory_cases.yaml"

_ERRORS = {
    "RuntimeError": RuntimeError,
    "TimeoutError": TimeoutError,
    "ValueError": ValueError,
}
_CANNED = {
    SeedData._OFF_TOPIC_REPLY: "off_topic",
    SeedData._NO_NEWS_FOUND_REPLY: "no_results",
    SEARCH_UNAVAILABLE_MESSAGE: "search_unavailable",
    SUMMARY_FAILED_MESSAGE: "summary_failed",
}
# keys check() understands; anything else in a case's expect / outcome is reported as an error
EXPECT_KEYS = frozenset(
    {
        "nodes",
        "tool_calls",
        "absent_tools",
        "llm_calls",
        "arg_checks",
        "llm_input_checks",
        "outcome",
    }
)
OUTCOME_KEYS = frozenset(
    {
        "http_status",
        "status",
        "error",
        "summary_name",
        "articles",
        "canned",
        "summary",
        "not_canned",
        "no_internal_leak",
        "articles_ids",
    }
)
_ERROR_NAMES = {
    SEARCH_UNAVAILABLE_MESSAGE: "search_unavailable",
    QUERY_FAILED_MESSAGE: "query_failed",
}


# --------------------------------------------------------------------------- loading
def load_cases(
    path: Path = CASES_FILE,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Read the trajectory cases file: (named node paths, list of cases)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["paths"], data["cases"]


# --------------------------------------------------------------------------- fakes
class ScriptedLLM:
    """Returns the case's llm_script in order; records which prompt each call came from."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        """script: the case's llm_script, one {classify|summarize: reply} step per expected call."""
        self._script = list(script)
        self.calls: list[str] = []
        self.inputs: list[list[BaseMessage]] = []
        self.runnable = RunnableLambda(self._respond)

    def _respond(self, prompt: PromptValue) -> AIMessage:
        """Return the next scripted reply; raise the scripted error or AssertionError on a mismatch."""
        messages = prompt.to_messages()
        system = str(messages[0].content) if messages else ""
        tag = "classify" if "news query analyser" in system else "summarize"
        self.calls.append(tag)
        self.inputs.append(messages)
        if not self._script:
            raise AssertionError(f"unexpected {tag} LLM call: llm_script exhausted")
        step = self._script.pop(0)
        if tag not in step:
            raise AssertionError(
                f"llm_script expected {list(step)} but the graph made a {tag} call"
            )
        reply = step[tag]
        if isinstance(reply, dict) and "raise" in reply:
            raise _ERRORS[reply["raise"]](f"scripted {tag} failure")
        return AIMessage(content=reply if isinstance(reply, str) else json.dumps(reply))


def mcp_from_case(spec: dict[str, Any] | None) -> FakeNewsMcp:
    """Build a FakeNewsMcp from a case's `mcp` block (category_hits, refs, docs, fail, crash)."""
    spec = spec or {}
    refs = [
        {"article_id": k, "score": float(v)}
        for item in spec.get("refs", [])
        for k, v in item.items()
    ]
    docs: dict[str, dict[str, Any]] = {}
    for item in spec.get("docs", []):
        if isinstance(item, str):
            docs[item] = make_doc(item)
        else:
            for doc_id, opts in item.items():
                docs[doc_id] = make_doc(
                    doc_id, content="x" * int(opts.get("content_len", 9))
                )
    return FakeNewsMcp(
        category_hits=int(spec.get("category_hits", 3)),
        refs=refs,
        docs=docs,
        fail=set(spec.get("fail", [])),
        crash=set(spec.get("crash", [])),
    )


# --------------------------------------------------------------------------- one run
@dataclass
class Trajectory:
    """What one run did: node order, MCP tool calls, LLM calls and inputs, and the outcome."""

    nodes: list[str] = field(default_factory=list)
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    llm_calls: list[str] = field(default_factory=list)
    llm_inputs: list[list[BaseMessage]] = field(default_factory=list)
    outcome: dict[str, Any] = field(default_factory=dict)


def outcome_of(response: Any) -> dict[str, Any]:
    """The fields cases assert on, from a QuerySuccessResponse or QueryFailedResponse.

    `_raw` keeps the whole JSON body for the no_internal_leak check.
    """
    body = response.model_dump(mode="json")
    if isinstance(response, QueryFailedResponse):
        return {
            "http_status": response.error_code,
            "status": response.status,
            "error": _ERROR_NAMES.get(response.error, response.error),
            "_raw": json.dumps(body),
        }
    return {
        "http_status": 200,
        "status": response.status,
        "summary_name": _CANNED.get(response.summary),
        "articles": body["articles"],
        "article_ids": [a["title"].removeprefix("Title ") for a in body["articles"]],
        "_raw": json.dumps(body),
    }


async def run_trajectory(
    query: str, llm_script: list[dict[str, Any]], mcp: dict[str, Any] | None
) -> Trajectory:
    """Run A: the app's run_graph (outcome, tool calls, LLM calls). Run B, fresh fakes: node order."""
    llm, fake = ScriptedLLM(llm_script), mcp_from_case(mcp)
    graph = await build(InMemorySaver(), llm.runnable, fake.client())  # type: ignore[arg-type]
    response = await run_graph(NewsInput(query=query), graph)

    llm_b, fake_b = ScriptedLLM(llm_script), mcp_from_case(mcp)
    graph_b = await build(InMemorySaver(), llm_b.runnable, fake_b.client())  # type: ignore[arg-type]
    nodes: list[str] = []
    config: RunnableConfig = {"configurable": {"thread_id": uuid4().hex}}
    # The failure itself is asserted through run A's outcome; here only the node order matters.
    with contextlib.suppress(Exception):
        async for update in graph_b.astream(
            NewsInput(query=query), config, stream_mode="updates"
        ):
            nodes.extend(update.keys())

    return Trajectory(
        nodes, list(fake.calls), list(llm.calls), list(llm.inputs), outcome_of(response)
    )


# --------------------------------------------------------------------------- matching
def _subset(expected: Any, actual: Any) -> bool:
    """True when every key in expected is in actual with a matching value (recursive)."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            k in actual and _subset(v, actual[k]) for k, v in expected.items()
        )
    return bool(expected == actual)


def _call_matches(exp: Any, act: tuple[str, dict[str, Any]]) -> bool:
    """True when an expected call (tool name, or {tool, args}) matches an actual (tool, args) call."""
    if isinstance(exp, str):
        return exp == act[0]
    return bool(exp["tool"] == act[0] and _subset(exp.get("args", {}), act[1]))


def _any_order(
    expected: list[Any], actual: list[Any], same: Callable[[Any, Any], bool]
) -> bool:
    """Same multiset: equal length and each expected item matches its own actual item."""
    if len(expected) != len(actual):
        return False
    unused = list(actual)
    for e in expected:
        hit = next((i for i, a in enumerate(unused) if same(e, a)), None)
        if hit is None:
            return False
        unused.pop(hit)
    return True


def _match(
    kind: str, expected: list[Any], actual: list[Any], same: Callable[[Any, Any], bool]
) -> bool:
    """Compare a trajectory list with the case's expectation (see the match types in the cases file).

    exact: same items, same order. in_order: expected is an ordered subsequence of
    actual. any_order: same multiset, matched with `same` (so tool args count too).
    Raises ValueError for an unknown match type.
    """
    if kind == "exact":
        return len(expected) == len(actual) and all(
            same(e, a) for e, a in zip(expected, actual)
        )
    if kind == "in_order":
        it = iter(actual)
        return all(any(same(e, a) for a in it) for e in expected)
    if kind == "any_order":
        return _any_order(expected, actual, same)
    raise ValueError(f"unknown match type {kind}")


def _dotted(obj: Any, path: str) -> Any:
    """Follow a dotted path ("user_query.categories") into nested dicts."""
    for part in path.split("."):
        obj = obj[part]
    return obj


def check(
    expect: dict[str, Any], traj: Trajectory, paths: dict[str, list[str]]
) -> list[str]:
    """Every failed expectation as a readable line; empty list = case passes.

    An unknown key in `expect` or `expect.outcome` (e.g. a typo such as `tool_call`) is
    itself an error, so a mistyped expectation fails the case instead of being skipped.
    """
    errors: list[str] = [
        f"expect: unknown key {key!r} (known: {sorted(EXPECT_KEYS)})"
        for key in expect
        if key not in EXPECT_KEYS
    ]
    errors += [
        f"outcome: unknown key {key!r} (known: {sorted(OUTCOME_KEYS)})"
        for key in expect.get("outcome", {})
        if key not in OUTCOME_KEYS
    ]

    if "nodes" in expect:
        want = expect["nodes"]["value"]
        want = paths[want] if isinstance(want, str) else want
        if not _match(expect["nodes"]["match"], want, traj.nodes, lambda e, a: e == a):
            errors.append(
                f"nodes: expected {want} ({expect['nodes']['match']}), got {traj.nodes}"
            )

    if "tool_calls" in expect:
        want = expect["tool_calls"]["value"]
        if not _match(
            expect["tool_calls"]["match"], want, traj.tool_calls, _call_matches
        ):
            errors.append(f"tool_calls: expected {want}, got {traj.tool_calls}")

    for tool in expect.get("absent_tools", []):
        if any(t == tool for t, _ in traj.tool_calls):
            errors.append(f"absent_tools: {tool} was called")

    if "llm_calls" in expect:
        want = expect["llm_calls"]["value"]
        if not _match(
            expect["llm_calls"]["match"], want, traj.llm_calls, lambda e, a: e == a
        ):
            errors.append(f"llm_calls: expected {want}, got {traj.llm_calls}")

    for ac in expect.get("arg_checks", []):
        calls = [args for t, args in traj.tool_calls if t == ac["tool"]]
        got = _dotted(calls[0], ac["path"]) if calls else "<no call>"
        if got != ac["equals"]:
            errors.append(
                f"arg_checks: {ac['tool']}.{ac['path']} = {got!r}, expected {ac['equals']!r}"
            )

    for lc in expect.get("llm_input_checks", []):
        idx = [i for i, tag in enumerate(traj.llm_calls) if tag == lc["llm_call"]]
        text = "\n".join(str(m.content) for m in traj.llm_inputs[idx[0]]) if idx else ""
        if "max_content_chars" in lc:
            n = int(lc["max_content_chars"])
            if "x" * (n + 1) in text or "x" * n not in text:
                errors.append(f"llm_input_checks: content not cut at {n} chars")
        for title in lc.get("titles", []):
            if title not in text:
                errors.append(
                    f"llm_input_checks: title {title!r} missing from the {lc['llm_call']} input"
                )

    out = expect.get("outcome", {})
    o = traj.outcome
    for key, want in out.items():
        if key == "not_canned":
            ok = o.get("summary_name") is None
        elif key == "canned" or key == "summary":
            ok = o.get("summary_name") == want
        elif key == "no_internal_leak":
            ok = not any(
                s in o["_raw"] for s in ("elasticsearch", "Traceback", "es host")
            )
        elif key == "articles_ids":
            ok = o.get("article_ids") == want
        else:
            ok = o.get(key) == want
        if not ok:
            errors.append(
                f"outcome.{key}: expected {want!r}, got {o.get(key, o.get('summary_name'))!r}"
            )
    return errors
