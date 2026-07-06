"""The serial dictator rule for temporal (perpetual) approval voting."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any, Generic, Protocol, TypeVar

Voter = TypeVar("Voter")
Alternative = TypeVar("Alternative")


class ApprovalProfile(Protocol[Voter]):
    """Shape of a single round's approval ballots, as produced by Lackner's
    synthetic data generators (see profiles.ApprovalProfile)."""

    voters: Sequence[Voter]
    approval_sets: Mapping[Voter, Any]


class SerialDictator(Generic[Voter, Alternative]):
    """Callable serial dictator rule.

    Fixes a permutation of the n voters. Each call advances through T rounds
    of an instance; in round t, the voter at position t (mod n) of the
    permutation is the dictator, and the winner is chosen uniformly at random
    from that voter's approved alternatives.
    """

    def __init__(
        self,
        voters: Sequence[Voter],
        permutation: Sequence[Voter] | None = None,
        random_state: random.Random | None = None,
    ) -> None:
        voters = list(voters)
        self._random_state = random_state if random_state is not None else random.Random()
        if permutation is None:
            permutation = voters.copy()
            self._random_state.shuffle(permutation)
        elif set(permutation) != set(voters):
            raise ValueError("permutation must contain exactly the given voters")
        self.voters = voters
        self.permutation = list(permutation)
        self.round = 0

    def __call__(self, instance: Sequence[ApprovalProfile[Voter]]) -> list[Alternative]:
        winners: list[Alternative] = []
        for profile in instance:
            dictator = self.permutation[self.round % len(self.permutation)]
            self.round += 1
            winners.append(self._random_state.choice(list(profile.approval_sets[dictator])))
        return winners
