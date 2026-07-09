"""Convert real .tsoi datasets to JSONL, as a one-time batch preprocessing
step.

Parses a directory of .tsoi files (one round per file) once and saves the
result to a single JSONL file, so that experiment runners can load
pre-parsed JSON instead of re-parsing raw .tsoi files (and paying their
parsing cost) on every run. Fully self-contained: uses only functions
defined in this module.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def tsoi_to_dict(path: Path | str) -> dict:
    try:
        with open(path) as file:
            lines = file.readlines()
    except Exception as e:
        print(f"Error reading file '{path}': {e}", file=sys.stderr)

    try:
        num_candidates = int(lines[0].split(",")[0])
    except Exception as e:
        print(f"Error parsing number of candidates from file '{path}': {e}", file=sys.stderr)

    try:
        num_voters = int(lines[num_candidates + 1].split(",")[0])
    except Exception as e:
        print(f"Error parsing number of voters from file '{path}': {e}", file=sys.stderr)

    try:
        candidate_lines = lines[1 : num_candidates + 1]
        candidate_id_strs = [line.split(",")[0] for line in candidate_lines]
        candidates = [int(candidate_id) for candidate_id in candidate_id_strs]
        candidates.sort()

        approval_lines = lines[num_candidates + 2 :]
        voters = [line[: line.find(":")] for line in approval_lines]
        stripped_lines = [line[line.find(":") + 1 :] for line in approval_lines]
        token_rows = [line.split(",") for line in stripped_lines]
        token_rows = [row[1:] for row in token_rows]  # drop leading per-voter metadata field
        token_rows = [
            [token[: token.find("[")] if "[" in token else token for token in row]
            for row in token_rows
        ]
        approvals = [[int(token) for token in row] for row in token_rows]

    except Exception as e:
        print(f"Error parsing approvals fomr file '{path}': {e}", file=sys.stderr)

    if len(approvals) != num_voters:
        raise Exception(
            f"Expected {num_voters} voter approvals but got {len(approvals)} in file '{path}'"
        )

    return {
        "voters": voters,
        "cands": candidates,
        "approval_sets": {voters[i]: approval_set for i, approval_set in enumerate(approvals)},
    }


def _fill_missing_voters(rounds: list[dict]) -> list[dict]:
    """Reconcile voters across rounds.

    Not every voter necessarily appears in every round (e.g. a country
    that didn't participate that year). Uses the union of voters seen
    across all rounds; a voter absent from a round is treated as
    indifferent between that round's candidates, i.e. as approving all
    of them, rather than none.
    """
    all_voters: list = []
    seen_voters: set = set()
    for round_data in rounds:
        for voter in round_data["voters"]:
            if voter not in seen_voters:
                seen_voters.add(voter)
                all_voters.append(voter)

    return [
        {
            "voters": all_voters,
            "cands": round_data["cands"],
            "approval_sets": {
                voter: round_data["approval_sets"].get(voter, round_data["cands"])
                for voter in all_voters
            },
        }
        for round_data in rounds
    ]


def tsoi_dir_to_json(directory: Path | str, output_path: Path | str) -> None:
    """Convert every .tsoi file in directory (one round per file, sorted
    chronologically) into a single JSONL file at output_path.

    The first line is a metadata record: {"T", "voters", "candidates"}
    (T = number of rounds, voters = the reconciled set of all voters
    across every round, candidates = the union of candidates across
    every round). Each following line is one round's record (see
    tsoi_to_dict), in order, with a "round" index added.

    Errors are caught and reported as human-readable messages on stderr
    rather than raised.
    """
    try:
        paths = sorted(Path(directory).iterdir())
    except Exception as e:
        print(f"Error reading directory '{directory}': {e}", file=sys.stderr)
        return

    rounds = []
    for path in paths:
        try:
            rounds.append(tsoi_to_dict(path))
        except Exception as e:
            print(f"Error parsing file '{path}': {e}", file=sys.stderr)

    if len(rounds) != len(paths):
        print(
            f"Error: expected {len(paths)} rounds but parsed {len(rounds)} in '{directory}'",
            file=sys.stderr,
        )
        return

    rounds = _fill_missing_voters(rounds)

    metadata = {
        "T": len(rounds),
        "voters": rounds[0]["voters"],
        "candidates": sorted({cand for round_data in rounds for cand in round_data["cands"]}),
    }

    try:
        with Path(output_path).open("w") as f:
            f.write(json.dumps(metadata) + "\n")
            for t, round_data in enumerate(rounds):
                record = {"round": t, **round_data}
                f.write(json.dumps(record) + "\n")
    except OSError as e:
        print(f"Error writing jsonl to '{output_path}': {e}", file=sys.stderr)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a directory of .tsoi files into a single JSONL file."
    )
    parser.add_argument("directory", type=Path, help="directory of .tsoi files, one round per file")
    parser.add_argument("output", type=Path, help="path to write the resulting .jsonl file")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    tsoi_dir_to_json(args.directory, args.output)
