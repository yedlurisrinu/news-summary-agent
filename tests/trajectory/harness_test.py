"""
tests/trajectory/harness_test.py

The trajectory harness's own checking logic (evals/trajectory/harness.py), no graph runs:
  - an unknown key in expect / outcome fails the case (a typo is never silently skipped)
  - any_order is a true multiset match that also compares tool args
  - the TRJ-021 run_2 expectation catches wrong mget ids / articles
"""

from __future__ import annotations

from evals.trajectory.harness import (
    EXPECT_KEYS,
    OUTCOME_KEYS,
    Trajectory,
    _match,
    check,
    load_cases,
)

PATHS, CASES = load_cases()
BY_ID = {c["id"]: c for c in CASES}


def _same_call(e, a) -> bool:
    from evals.trajectory.harness import _call_matches

    return _call_matches(e, a)


class TestUnknownKeys:
    def test_typo_in_expect_fails(self) -> None:
        errors = check(
            {"tool_call": {"match": "exact", "value": []}}, Trajectory(), PATHS
        )
        assert errors and "unknown key 'tool_call'" in errors[0]

    def test_typo_in_outcome_fails(self) -> None:
        traj = Trajectory(outcome={"http_status": 200})
        errors = check({"outcome": {"http_stauts": 200}}, traj, PATHS)
        assert any("unknown key 'http_stauts'" in e for e in errors)

    def test_every_case_uses_known_keys_only(self) -> None:
        for case in CASES:
            if "expect" in case and "query" in case and "applies_to" not in case:
                expect = case["expect"]
                assert set(expect) <= EXPECT_KEYS, case["id"]
                assert set(expect.get("outcome", {})) <= OUTCOME_KEYS, case["id"]


class TestAnyOrder:
    def test_same_names_different_args_fail(self) -> None:
        expected = [{"tool": "mget", "args": {"ids": ["a1"]}}]
        actual = [("mget", {"ids": ["zz"]})]
        assert not _match("any_order", expected, actual, _same_call)

    def test_reordered_calls_pass(self) -> None:
        expected = [{"tool": "mget", "args": {"ids": ["a1"]}}, "article_search"]
        actual = [("article_search", {}), ("mget", {"ids": ["a1"]})]
        assert _match("any_order", expected, actual, _same_call)

    def test_multiset_counts(self) -> None:
        assert not _match("any_order", ["mget", "mget"], [("mget", {})], _same_call)
        assert not _match(
            "any_order", ["mget"], [("mget", {}), ("mget", {})], _same_call
        )


class TestTrj021Expectation:
    def _run_2(self, ids: list[str]) -> Trajectory:
        return Trajectory(
            tool_calls=[("mget", {"ids": ids})],
            outcome={"article_ids": ids, "http_status": 200},
        )

    def test_passes_on_own_articles(self) -> None:
        assert (
            check(BY_ID["TRJ-021"]["expect"]["run_2"], self._run_2(["s1"]), PATHS) == []
        )

    def test_fails_on_leaked_articles(self) -> None:
        errors = check(
            BY_ID["TRJ-021"]["expect"]["run_2"], self._run_2(["f1", "s1"]), PATHS
        )
        assert any("arg_checks" in e for e in errors)
        assert any("articles_ids" in e for e in errors)
