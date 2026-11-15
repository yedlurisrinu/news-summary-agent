"""Scoring rules of the classify component suite, on hand-made outputs (no LLM)."""

from __future__ import annotations

import pytest

from evals.component.classify_suite import (
    Output,
    Taxonomy,
    evaluate_gate,
    jaccard,
    score_case,
)

pytestmark = pytest.mark.anyio

TAX = Taxonomy(
    allowed=frozenset(
        {"technology", "politics", "sports", "business", "finance", "ai", "world"}
    ),
    alias_to_canonical={"tech": "technology", "fin": "finance"},
)


def _out(is_news: bool, cats: list[str], latest: bool) -> Output:
    parsed = {"is_news_query": is_news, "categories": cats, "fetch_latest": latest}
    return Output(raw=str(parsed), parsed=parsed, json_valid=True)


def _case(
    must: list[str],
    may: list[str],
    is_news: bool = True,
    latest: bool = False,
    **extra: object,
) -> dict:
    return {
        "id": "C",
        "input": {"query": "q"},
        "expected": {
            "is_news_query": is_news,
            "categories_must": must,
            "categories_may": may,
            "fetch_latest": latest,
        },
        **extra,
    }


def test_alias_counts_as_canonical_category() -> None:
    r = score_case(_case(["technology"], ["ai"]), [_out(True, ["tech"], False)], TAX)
    assert r.checks["categories_must"] and r.checks["categories_in_bounds"] and r.passed


def test_category_outside_must_and_may_is_out_of_bounds() -> None:
    r = score_case(
        _case(["technology"], []), [_out(True, ["technology", "sports"], False)], TAX
    )
    assert r.checks["categories_in_bounds"] is False and not r.passed


def test_missing_must_category_fails() -> None:
    r = score_case(
        _case(["technology", "politics"], []), [_out(True, ["technology"], False)], TAX
    )
    assert r.checks["categories_must"] is False


def test_invalid_json_fails_every_check() -> None:
    bad = Output(
        raw="Sure! news about AI", parsed=None, json_valid=False, error="parse"
    )
    r = score_case(_case(["ai"], []), [bad], TAX)
    assert r.checks == {
        "json_valid": False,
        "is_news_query": False,
        "fetch_latest": False,
        "categories_must": False,
        "categories_in_bounds": False,
    }


def test_judged_criterion_is_recorded_not_scored() -> None:
    case = _case([], ["technology"], judge={"criteria": ["categories_reasonable"]})
    r = score_case(case, [_out(True, ["technology"], False)], TAX)
    assert r.checks["judged:categories_reasonable"] is None and r.passed


def test_stability_flags_a_flip() -> None:
    case = {"id": "S", "input": {"query": "q", "repeat": 3}}
    outs = [
        _out(True, ["ai"], True),
        _out(True, ["ai"], False),
        _out(True, ["ai"], True),
    ]
    r = score_case(case, outs, TAX)
    assert r.checks["is_news_query_stable"] and r.checks["fetch_latest_stable"] is False


def test_jaccard_of_two_empty_sets_is_one() -> None:
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a", "b"}, {"a"}) == 0.5

async def test_gate_verdicts() -> None:
    assert (await evaluate_gate("g", 12, 12, 1.0)).verdict == "met"
    assert (await evaluate_gate("g", 11, 12, 1.0)).verdict == "missed"
    assert (
        await evaluate_gate("g", 29, 29, 0.90)
    ).verdict == "inconclusive"  # lower 0.883 < 0.90
    assert (await evaluate_gate("g", 100, 100, 0.90)).verdict == "met"
    assert (await evaluate_gate("g", 5, 20, 0.90)).verdict == "missed"
    assert (await evaluate_gate("g", 0, 0, 0.90)).verdict == "no_data"