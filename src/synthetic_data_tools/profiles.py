"""Approval profiles for perpetual voting rules.

Ported from Martin Lackner's perpetual voting codebase (profiles.py).
Logic is unchanged; only the Python 2/3 compatibility shims
(collections.Mapping, future.utils.iteritems) are replaced with their
Python 3.11 equivalents.
"""

from __future__ import annotations

import collections.abc
import copy

import numpy.random as random
from scipy.spatial.distance import euclidean


# approval profile
class ApprovalProfile:
    def __init__(self, voters, cands, approval_sets):
        self.voters = voters
        if isinstance(approval_sets, collections.abc.Mapping):
            self.approval_sets = approval_sets
        elif isinstance(approval_sets, list):
            assert len(approval_sets) == len(voters)
            self.approval_sets = {}
            for i in range(len(voters)):
                self.approval_sets[voters[i]] = approval_sets[i]
        else:
            raise Exception("type of approval_sets neither dict nor list")
        self.cands = cands
        # Membership is checked against sets rather than voters/cands
        # directly -- those are kept as lists (order matters elsewhere),
        # but checking `in` on a list is O(n); with large candidate
        # counts and approval sets, checking every approved candidate
        # against a list made this validation loop quadratic.
        voters_set = set(voters)
        cands_set = set(cands)
        for v, appr in self.approval_sets.items():
            if v not in voters_set:
                raise Exception(
                    str(v) + " is not a valid voter; " + "voters are " + str(voters) + "."
                )
            for c in appr:
                if c not in cands_set:
                    raise Exception(
                        str(c)
                        + " is not a valid candidate; "
                        + "candidates are "
                        + str(cands)
                        + "."
                    )

    def __str__(self):
        return (
            f"Profile with {len(self.voters)} votes and {len(self.cands)} candidates: "
            + ", ".join(map(str, self.approval_sets.values()))
        )

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        voters = list(self.voters)
        approvals_sets = copy.deepcopy(self.approval_sets)
        cands = list(self.cands)
        return ApprovalProfile(voters, cands, approvals_sets)

    def has_empty_sets(self):
        return any(len(appr) == 0 for appr in self.approval_sets.values())


# uniformly random profile:
# voters' approval sets have a size given by dict approval_set_sizes
def uniformly_random_profile(voters, cands, approval_set_sizes):
    approval_sets = {}
    for v in voters:
        approval_sets[v] = set(random.choice(cands, approval_set_sizes[v], replace=False))
    return ApprovalProfile(voters, cands, approval_sets)


# create approval profile from 2d coordinates (Euclidean distance)
def approvalprofile_from_2d_euclidean(voters, cands, voter_points, cand_points, threshold):
    approval_sets = {}
    for v in voters:
        distances = {c: euclidean(voter_points[v], cand_points[c]) for c in cands}
        mindist = min(distances.values())
        approval_sets[v] = [c for c in cands if distances[c] <= mindist * threshold]
    return ApprovalProfile(voters, cands, approval_sets)
