"""Generation of temporal (perpetual) voting instances.

Core generation logic ported from Martin Lackner's perpetual voting
codebase (experiments/experiments.py: generate_instances), with the
per-spec pickle caching removed since that is experiment-management
bookkeeping, not data generation.
"""

from __future__ import annotations

import random

from .points import generate_2d_points
from .profiles import ApprovalProfile, approvalprofile_from_2d_euclidean


def generate_instance(
    n: int,
    m: int,
    T: int,
    sigma: float,
    voter_point_mode: str,
    cand_point_mode: str,
    approval_threshold: float,
    seed: int | None = None,
) -> list[ApprovalProfile]:
    """Generate a single temporal voting instance: n voters, T rounds,
    up to m alternatives per round, positioned in 2d Euclidean space.

    seed makes point generation reproducible: the same seed (with the
    same other parameters) always produces the same instance, since a
    single random.Random(seed) drives voter point generation followed by
    every round's candidate point generation, in order.
    """
    random_state = random.Random(seed)
    voters = list(range(n))
    cands = list(range(m))
    voter_points = generate_2d_points(voters, voter_point_mode, sigma, random_state)

    history = []
    for _ in range(T):
        cand_points = generate_2d_points(cands, cand_point_mode, sigma, random_state)
        prof = approvalprofile_from_2d_euclidean(
            voters, cands, voter_points, cand_points, approval_threshold
        )
        history.append(prof)
    return history
