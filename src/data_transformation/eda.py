"""Exploratory data analysis for converted real-data JSONL datasets.

Operates on the JSONL files produced by tsoi_to_json.tsoi_dir_to_json:
a metadata record on the first line, followed by one record per round
({"round", "voters", "cands", "approval_sets"}).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load_rounds(path: Path | str) -> list[dict] | None:
    """Load the per-round records from a jsonl dataset, skipping the
    metadata line. Returns None if the file could not be read or parsed.
    """
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError as e:
        print(f"Error reading '{path}': {e}", file=sys.stderr)
        return None

    try:
        return [json.loads(line) for line in lines[1:]]
    except json.JSONDecodeError as e:
        print(f"Error decoding '{path}': {e}", file=sys.stderr)
        return None


def candidates_same_each_round(path: Path | str) -> bool | None:
    """Check whether every round in the jsonl dataset at path has the
    exact same set of candidates. Returns None if the dataset could not
    be loaded.
    """
    rounds = _load_rounds(path)
    if rounds is None:
        return None
    if not rounds:
        return True

    first_cands = set(rounds[0]["cands"])
    return all(set(round_data["cands"]) == first_cands for round_data in rounds)


def voters_same_each_round(path: Path | str) -> bool | None:
    """Check whether every round in the jsonl dataset at path has the
    exact same set of voters. Returns None if the dataset could not be
    loaded.
    """
    rounds = _load_rounds(path)
    if rounds is None:
        return None
    if not rounds:
        return True

    first_voters = set(rounds[0]["voters"])
    return all(set(round_data["voters"]) == first_voters for round_data in rounds)
