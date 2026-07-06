"""JSONL encoding of temporal voting instances, for reproducible storage."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TypeVar

Voter = TypeVar("Voter")
Alternative = TypeVar("Alternative")


class ApprovalProfile(Protocol[Voter, Alternative]):
    voters: Sequence[Voter]
    cands: Sequence[Alternative]
    approval_sets: Mapping[Voter, Collection[Alternative]]


def save_profile_jsonl(instance: Sequence[ApprovalProfile[Any, Any]], path: str | Path) -> None:
    """Encode a temporal voting instance as JSONL (one JSON object per
    round) and save it to `path`."""
    with Path(path).open("w") as f:
        for t, profile in enumerate(instance):
            record = {
                "round": t,
                "voters": list(profile.voters),
                "cands": list(profile.cands),
                "approval_sets": {
                    str(voter): sorted(profile.approval_sets[voter], key=str)
                    for voter in profile.voters
                },
            }
            f.write(json.dumps(record) + "\n")
