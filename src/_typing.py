"""Shared structural types for temporal voting instances.

A single canonical Protocol describing the shape of one round's approval
ballots (voters, cands, approval_sets), used across the codebase
(encoding, printing, voting rules) without creating a hard dependency on
any one concrete implementation -- synthetic_data_tools.profiles's
ApprovalProfile satisfies this shape, but so would anything else with the
same attributes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeVar

Voter = TypeVar("Voter")
Alternative = TypeVar("Alternative")


class ApprovalProfile(Protocol[Voter, Alternative]):
    """Structural shape of a single round's approval ballots."""

    voters: Sequence[Voter]
    cands: Sequence[Alternative]
    approval_sets: Mapping[Voter, Any]
