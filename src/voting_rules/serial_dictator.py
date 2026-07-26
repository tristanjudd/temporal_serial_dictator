"""The serial dictator rule for temporal (perpetual) approval voting."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Generic

from .._typing import Alternative, ApprovalProfile, Voter


class SerialDictator(Generic[Voter, Alternative]):
    """Callable serial dictator rule.

    Fixes a permutation of the n voters. Each call advances through T rounds
    of an instance; in round t, the voter at position t (mod n) of the
    permutation is the dictator, and the winner is chosen uniformly at random
    from that voter's approved alternatives -- or, if the dictator's
    approval set is empty (indifferent that round), uniformly at random
    from all of that round's alternatives instead.

    By default the permutation is just the given voter order (the identity
    permutation) -- deterministic, and isomorphic to any other fixed
    permutation, since voter labels carry no meaning beyond identity. Call
    permute_voters() explicitly to draw a fresh random permutation instead
    (e.g. for a randomized-serial-dictator variant).

    seed makes both sources of randomness reproducible: the winner choice
    made on every call (random.Random.choice over the dictator's approved
    alternatives) and, if permute_voters() is ever used, the permutation
    shuffle. The same seed always produces the same sequence of choices
    across repeated calls to the same SerialDictator instance.
    """

    def __init__(
        self,
        voters: Sequence[Voter],
        permutation: Sequence[Voter] | None = None,
        seed: int | None = None,
    ) -> None:
        voters = list(voters)
        self._random_state = random.Random(seed)
        if permutation is None:
            permutation = voters.copy()
        elif set(permutation) != set(voters):
            raise ValueError("permutation must contain exactly the given voters")
        self.voters = voters
        self.permutation = list(permutation)
        self.round = 0

    def reset(self) -> None:
        """Restart the round counter, keeping the current permutation.

        Use this between independent runs that should share the same
        voter ordering (e.g. repeated runs over fresh synthetic
        instances), so each run's rounds start at permutation[0] again
        instead of continuing on from the previous run.
        """
        self.round = 0

    def permute_voters(self) -> None:
        """Draw a fresh random permutation over the voters, and reset the
        round counter so the next call starts from the beginning of it."""
        self._random_state.shuffle(self.permutation)
        self.reset()

    def __call__(
        self, instance: Sequence[ApprovalProfile[Voter, Alternative]]
    ) -> list[Alternative]:
        winners: list[Alternative] = []
        for profile in instance:
            dictator = self.permutation[self.round % len(self.permutation)]
            self.round += 1
            approved = profile.approval_sets[dictator]
            choices = list(approved) if approved else list(profile.cands)
            winners.append(self._random_state.choice(choices))
        return winners
