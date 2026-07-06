"""JSONL encoding of temporal voting instances, for reproducible storage."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TypeVar

Voter = TypeVar("Voter")
Alternative = TypeVar("Alternative")


class ApprovalProfile(Protocol[Voter, Alternative]):
    voters: Sequence[Voter]
    cands: Sequence[Alternative]
    approval_sets: Mapping[Voter, Any]


def save_profile_jsonl(instance: Sequence[ApprovalProfile[Any, Any]], path: str | Path) -> None:
    """Encode a temporal voting instance as JSONL (one JSON object per
    round) and save it to `path`.

    Errors are caught and reported as human-readable messages on stderr
    rather than raised.
    """
    try:
        lines = [
            json.dumps(
                {
                    "round": t,
                    "voters": list(profile.voters),
                    "cands": list(profile.cands),
                    "approval_sets": {
                        str(voter): sorted(profile.approval_sets[voter], key=str)
                        for voter in profile.voters
                    },
                }
            )
            for t, profile in enumerate(instance)
        ]
    except KeyError as e:
        print(f"Error encoding profile: voter {e} has no approval set.", file=sys.stderr)
        return
    except TypeError as e:
        print(f"Error encoding profile: could not convert data to JSON ({e}).", file=sys.stderr)
        return

    try:
        Path(path).write_text("".join(line + "\n" for line in lines))
    except OSError as e:
        print(f"Error writing profile to '{path}': {e}.", file=sys.stderr)
