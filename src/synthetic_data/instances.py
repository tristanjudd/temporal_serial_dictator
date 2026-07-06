"""Generation of temporal (perpetual) voting instances.

Core generation logic ported from Martin Lackner's perpetual voting
codebase (experiments/experiments.py: generate_instances), with the
per-spec pickle caching removed since that is experiment-management
bookkeeping, not data generation.
"""

from __future__ import annotations

from synthetic_data.points import generate_2d_points
from synthetic_data.profiles import ApprovalProfile, approvalprofile_from_2d_euclidean


def generate_instance(
    n: int,
    m: int,
    T: int,
    sigma: float,
    voter_point_mode: str,
    cand_point_mode: str,
    approval_threshold: float,
) -> list[ApprovalProfile]:
    """Generate a single temporal voting instance: n voters, T rounds,
    up to m alternatives per round, positioned in 2d Euclidean space."""
    voters = list(range(n))
    cands = list(range(m))
    voter_points = generate_2d_points(voters, voter_point_mode, sigma)

    history = []
    for _ in range(T):
        cand_points = generate_2d_points(cands, cand_point_mode, sigma)
        prof = approvalprofile_from_2d_euclidean(
            voters, cands, voter_points, cand_points, approval_threshold
        )
        history.append(prof)
    return history
