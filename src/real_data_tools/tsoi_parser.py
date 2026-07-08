import os
import sys
from pathlib import Path

from ..synthetic_data_tools.profiles import ApprovalProfile


def parse_tsoi_directory(path: Path | str) -> list[ApprovalProfile]:
    try:
        filenames = sorted(os.listdir(path))
    except Exception as e:
        print(f"Error reading directory '{path}': {e}", file=sys.stderr)
        return []

    approval_sequence = []

    for filename in filenames:
        file_path = os.path.join(path, filename)
        try:
            approval_sequence.append(parse_tsoi(file_path))
        except Exception as e:
            print(f"Error parsing file '{file_path}': {e}", file=sys.stderr)

    if len(approval_sequence) != len(filenames):
        raise Exception(
            f"Expected {len(filenames)} approval profiles but got "
            f"{len(approval_sequence)} in directory '{path}'"
        )
    if any(profile.has_empty_sets() for profile in approval_sequence):
        raise Exception(f"Found an approval profile with empty approval sets in '{path}'")

    return _fill_missing_voters(approval_sequence)


def _fill_missing_voters(approval_sequence: list[ApprovalProfile]) -> list[ApprovalProfile]:
    """Reconcile voters across rounds.

    Not every voter necessarily appears in every round (e.g. a country
    that didn't participate that year). Uses the union of voters seen
    across all rounds; a voter absent from a round is treated as
    indifferent between that round's candidates, i.e. as approving all
    of them, rather than none.
    """
    all_voters = []
    seen_voters = set()
    for profile in approval_sequence:
        for voter in profile.voters:
            if voter not in seen_voters:
                seen_voters.add(voter)
                all_voters.append(voter)

    return [
        ApprovalProfile(
            voters=all_voters,
            cands=profile.cands,
            approval_sets={
                voter: profile.approval_sets.get(voter, profile.cands) for voter in all_voters
            },
        )
        for profile in approval_sequence
    ]


def parse_tsoi(path: Path | str) -> ApprovalProfile:
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

    approval_profile = ApprovalProfile(
        voters=voters,
        cands=candidates,
        approval_sets={voters[i]: approval_set for i, approval_set in enumerate(approvals)},
    )

    return approval_profile
