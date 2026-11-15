"""
evals/fixtures/fixture_capture_runner.py — capture e2e cases: run the app's graph against
the test instance and freeze each run as response.json + state.json (e2e_cases.yaml,
"How a case runs").

  uv run python -m evals.fixtures.fixture_capture_runner --only E2E-001   # smoke: one case
  uv run python -m evals.fixtures.fixture_capture_runner                  # all 28 cases

Per case: a fresh InMemorySaver, the graph compiled by the app's build(), one call to the
app's run_graph() (same response mapping as POST /query), then the final state read from
the only thread in that saver. A failing case is still a capture (its outcome is recorded);
a preflight failure stops the run before any case.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from langgraph.checkpoint.memory import InMemorySaver

from config.application_config import load_config as app_load_config
from config.logging_setup import setup_logging
from config.vault_secrets import load_secrets
from data.news_input import NewsInput
from data.query_failed_response import QueryFailedResponse
from evals.eval_config.load_eval_config import get_config, load_config
from evals.e2e.fixtures.fixture_capture_runtime import EvalRuntime, open_runtime
from evals.e2e.helpers.constants import EVAL_BASE_PATH, E2E_BASE_PATH
from graph.build_graph import build
from graph.run_graph_workflow import run_graph


# paths come from eval_config.yaml, so they are resolved after load_config() has run
def _cases_file() -> Path:
    """Absolute path of the e2e seed cases file (SEED_E2E_CASES_PATH)."""
    return EVAL_BASE_PATH.joinpath(get_config("SEED_E2E_CASES_PATH"))


def _fixtures() -> Path:
    """Absolute path of the e2e fixtures folder (FIXTURES_E2E_PATH)."""
    return E2E_BASE_PATH.joinpath(get_config("FIXTURES_E2E_PATH"))


def _runs() -> Path:
    """Absolute path of the capture-run records folder (CAPTURE_RUNS_PATH)."""
    return E2E_BASE_PATH.joinpath(get_config("CAPTURE_RUNS_PATH"))


def _sha12(path: Path) -> str:
    """First 12 hex characters of the file's SHA-256, used as a content version."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _git_rev() -> str:
    """Short git sha of HEAD for provenance, or "unknown" outside a git checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


async def capture_case(rt: EvalRuntime, case: dict[str, Any]) -> dict[str, Any]:
    """Run one case through the app's graph and freeze response.json + state.json.

    Uses a fresh InMemorySaver so the final state can be read back from its only thread.
    Returns the case's summary line for the run record: case_id, http_status, threads, elapsed_s.
    """
    saver = InMemorySaver()
    graph = await build(saver, rt.llm, rt.mcp_client)  # type: ignore[arg-type]  # build() is annotated AsyncPostgresSaver
    started = time.perf_counter()
    response = await run_graph(NewsInput(query=case["query"]), graph)
    elapsed = time.perf_counter() - started
    thread_ids = {
        t.config["configurable"]["thread_id"] async for t in saver.alist(None)
    }
    state: dict[str, Any] | None = None
    n_threads = len(thread_ids)
    if n_threads == 1:
        (thread_id,) = thread_ids
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        state = dict(snapshot.values)

    http_status = (
        response.error_code if isinstance(response, QueryFailedResponse) else 200
    )
    out = _fixtures() / case["id"] / rt.run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "response.json").write_text(
        json.dumps(
            {"http_status": http_status, "body": response.model_dump(mode="json")},
            indent=2,
            ensure_ascii=False,
        )
    )
    (out / "state.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False, default=str)
    )
    return {
        "case_id": case["id"],
        "http_status": http_status,
        "threads": n_threads,
        "elapsed_s": round(elapsed, 2),
    }


async def verify_frozen(rt: EvalRuntime, case_id: str) -> dict[str, Any]:
    """Smoke check: every returned article exists in the frozen index and was published before reference_now."""
    state = json.loads((_fixtures() / case_id / rt.run_id / "state.json").read_text())
    ids = [h["article_id"] for h in (state or {}).get("article_hits", [])]
    if not ids:
        return {"case_id": case_id, "articles": 0}
    r = await rt.es_client.post(
        f"/{rt.frozen_index}/_mget",
        json={"ids": ids},
        params={"_source": "published_at"},
    )
    r.raise_for_status()
    docs = r.json()["docs"]
    missing = [d["_id"] for d in docs if not d.get("found")]
    ref = datetime.fromisoformat(rt.meta["reference_now"])
    after_freeze = [
        d["_id"]
        for d in docs
        if d.get("found") and datetime.fromisoformat(d["_source"]["published_at"]) > ref
    ]
    return {
        "case_id": case_id,
        "articles": len(ids),
        "missing_in_frozen": missing,
        "published_after_reference_now": after_freeze,
    }


async def main(only: list[str] | None) -> None:
    """Capture every case (or only the given ids) into one run, then write capture_runs/<run_id>.json.

    The run record is written even when a case raises, with what was captured so far.
    """
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    cases_file = _cases_file()
    cases = yaml.safe_load(cases_file.read_text())["cases"]
    if only:
        cases = [c for c in cases if c["id"] in only]
    async with open_runtime(run_id) as rt:
        record: dict[str, Any] = {
            "run_id": run_id,
            "mode": "capture",
            "golden_set_version": {
                "e2e_cases_sha": _sha12(cases_file),
                "frozen_index": rt.frozen_index,
            },
            "reference_now": rt.meta["reference_now"],
            "offset_ms": rt.offset_ms,
            "retriever": {"backend": "lexical", "mcp_url": rt.mcp_url},
            "app_version": _git_rev(),
            "started_at": datetime.now(UTC).isoformat(),
            "cases": [],
        }
        try:
            for case in cases:
                result = await capture_case(rt, case)
                result["frozen_check"] = await verify_frozen(rt, case["id"])
                record["cases"].append(result)
                print(json.dumps(result))
        finally:
            record["ended_at"] = datetime.now(UTC).isoformat()
            runs = _runs()
            runs.mkdir(parents=True, exist_ok=True)
            (runs / f"{run_id}.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    setup_logging()
    load_secrets()
    load_config()
    app_load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", help="case ids, e.g. E2E-001")
    asyncio.run(main(p.parse_args().only))
