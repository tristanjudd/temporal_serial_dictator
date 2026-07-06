"""JSONL decoding of temporal voting instances, for reproducible storage."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ..synthetic_data.profiles import ApprovalProfile


def load_profile_jsonl(path: str | Path) -> list[ApprovalProfile] | None:
    """Read a JSONL-encoded temporal voting instance from `path` (as
    written by encoding.save_profile_jsonl) and reconstruct it as a list
    of ApprovalProfile objects, one per round.

    Errors are caught and reported as human-readable messages on stderr
    rather than raised; None is returned if decoding fails.
    """
    try:
        lines = Path(path).read_text().splitlines()
    except OSError as e:
        print(f"Error reading profile from '{path}': {e}.", file=sys.stderr)
        return None

    instance = []
    for i, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error decoding profile: line {i + 1} is not valid JSON ({e}).", file=sys.stderr)
            return None

        try:
            round_index = record["round"]
            voters = record["voters"]
            cands = record["cands"]
            raw_approval_sets = record["approval_sets"]
        except KeyError as e:
            print(f"Error decoding profile: line {i + 1} is missing field {e}.", file=sys.stderr)
            return None

        if round_index != i:
            print(
                f"Error decoding profile: line {i + 1} has round {round_index}, expected {i}.",
                file=sys.stderr,
            )
            return None

        voter_by_key = {str(voter): voter for voter in voters}
        approval_sets = {}
        for key, approved in raw_approval_sets.items():
            if key not in voter_by_key:
                print(
                    f"Error decoding profile: line {i + 1} has an approval set for "
                    f"unknown voter {key!r}.",
                    file=sys.stderr,
                )
                return None
            approval_sets[voter_by_key[key]] = approved

        try:
            instance.append(ApprovalProfile(voters, cands, approval_sets))
        except Exception as e:
            print(f"Error decoding profile: line {i + 1} is invalid ({e}).", file=sys.stderr)
            return None

    return instance


def load_decisions_json(path: str | Path) -> list[Any] | None:
    """Read a JSON-encoded decision sequence from `path` (as written by
    encoding.save_decisions_json).

    Errors are caught and reported as human-readable messages on stderr
    rather than raised; None is returned if decoding fails.
    """
    try:
        text = Path(path).read_text()
    except OSError as e:
        print(f"Error reading decision sequence from '{path}': {e}.", file=sys.stderr)
        return None

    try:
        decisions = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Error decoding decision sequence: not valid JSON ({e}).", file=sys.stderr)
        return None

    if not isinstance(decisions, list):
        print(
            "Error decoding decision sequence: expected a JSON array, got "
            f"{type(decisions).__name__}.",
            file=sys.stderr,
        )
        return None

    return decisions
