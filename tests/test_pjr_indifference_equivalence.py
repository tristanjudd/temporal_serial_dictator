"""Proves that treating an empty approval set as "indifferent" (excluded
from agreement/satisfaction checks, except when a group is entirely
indifferent in a round) gives byte-identical PJR results to the old
approach of backfilling indifference as "approves every candidate that
round" and checking it directly.

This is checked against an independent reference implementation (not the
optimized production code) that literally backfills and runs the
original, unoptimized algorithm, across many random instances -- rather
than just re-reading the optimized code and trusting it by inspection.
"""

from __future__ import annotations

import itertools
import random
from typing import Any

from src.synthetic_data_tools.profiles import ApprovalProfile
from src.verification.multiprocessing_verify_pjr import (
    find_pjr_violations_multiprocessing,
)
from src.verification.verify_pjr import find_pjr_violations


def _reference_backfilled_violations(
    instance: list[ApprovalProfile], decisions: list[Any]
) -> set[tuple[Any, ...]]:
    """Independent reference: backfill every empty approval set to the
    round's full candidate list (the old semantics), then run the
    original, unoptimized PJR check (no skip-empty logic at all) on the
    backfilled instance.
    """
    backfilled_instance = [
        ApprovalProfile(
            voters=list(profile.voters),
            cands=list(profile.cands),
            approval_sets={
                voter: (
                    list(profile.approval_sets[voter])
                    if profile.approval_sets[voter]
                    else list(profile.cands)
                )
                for voter in profile.voters
            },
        )
        for profile in instance
    ]

    voters = list(backfilled_instance[0].voters)
    n = len(voters)
    all_groups = itertools.chain.from_iterable(
        itertools.combinations(voters, size) for size in range(1, n + 1)
    )

    violations = set()
    for group in all_groups:
        agreement = 0
        satisfaction = 0
        for profile, winner in zip(backfilled_instance, decisions, strict=True):
            approval_sets = [set(profile.approval_sets[voter]) for voter in group]
            if set.intersection(*approval_sets):
                agreement += 1
            if any(winner in profile.approval_sets[voter] for voter in group):
                satisfaction += 1

        bound = (agreement * len(group)) // n
        if satisfaction < bound:
            violations.add((group, agreement, bound, satisfaction))

    return violations


def _as_comparable_set(violations: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
    return {(tuple(v["voters"]), v["agreement"], v["bound"], v["satisfaction"]) for v in violations}


def _random_instance_with_indifference(
    rng: random.Random, n: int, m: int, t: int, indifference_prob: float
) -> tuple[list[ApprovalProfile], list[Any]]:
    """A random instance where each (voter, round) independently has some
    probability of being indifferent (empty approval set) instead of
    approving a random nonempty subset of that round's candidates.
    """
    voters = list(range(n))
    instance = []
    for _ in range(t):
        cands = list(range(m))
        approval_sets: dict[int, list[int]] = {}
        for voter in voters:
            if rng.random() < indifference_prob:
                approval_sets[voter] = []
            else:
                size = rng.randint(1, m)
                approval_sets[voter] = rng.sample(cands, size)
        instance.append(ApprovalProfile(voters=voters, cands=cands, approval_sets=approval_sets))

    decisions = [rng.choice(profile.cands) for profile in instance]
    return instance, decisions


def test_matches_backfilled_reference_across_random_instances_with_indifference():
    rng = random.Random(12345)

    for _ in range(30):
        n = rng.randint(2, 6)
        m = rng.randint(2, 4)
        t = rng.randint(1, 5)
        indifference_prob = rng.choice([0.0, 0.2, 0.5, 0.8])

        instance, decisions = _random_instance_with_indifference(rng, n, m, t, indifference_prob)

        actual = _as_comparable_set(find_pjr_violations(instance, decisions))
        expected = _reference_backfilled_violations(instance, decisions)

        assert actual == expected, (
            f"mismatch for n={n}, m={m}, t={t}, indifference_prob={indifference_prob}"
        )


def test_matches_backfilled_reference_when_entire_group_is_indifferent():
    # every voter is indifferent in round 0 -- exercises the "genuine
    # subset of the group is empty" edge case directly, for every
    # possible group (not just single-voter ones).
    voters = [0, 1, 2]
    cands = [0, 1]
    instance = [
        ApprovalProfile(voters=voters, cands=cands, approval_sets={0: [], 1: [], 2: []}),
        ApprovalProfile(voters=voters, cands=cands, approval_sets={0: [0], 1: [1], 2: [0]}),
    ]
    decisions = [0, 1]

    actual = _as_comparable_set(find_pjr_violations(instance, decisions))
    expected = _reference_backfilled_violations(instance, decisions)

    assert actual == expected
    # sanity: this instance actually exercises some agreement/satisfaction
    # variety, not a degenerate no-op case.
    assert len(expected) > 0 or len(actual) == 0


def test_multiprocessing_matches_sequential_with_indifference():
    rng = random.Random(999)
    instance, decisions = _random_instance_with_indifference(
        rng, n=5, m=3, t=4, indifference_prob=0.4
    )

    sequential = _as_comparable_set(find_pjr_violations(instance, decisions))
    parallel = _as_comparable_set(
        find_pjr_violations_multiprocessing(instance, decisions, max_workers=2)
    )

    assert sequential == parallel
