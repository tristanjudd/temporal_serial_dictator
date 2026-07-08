"""Run a serial-dictator experiment on a real (natural) temporal voting
dataset.

Loads a temporal approval voting instance from a directory of .tsoi files
(one round per file, all sharing the same voters, via
tsoi_parser.parse_tsoi_directory), runs the serial dictator rule on it,
and saves the approval profile and decision sequence under
experiments/<dataset name>/run_<n>/, mirroring
synthetic_data_tools.run_synthetic_experiment's pipeline.

Unlike the synthetic experiment, T (rounds), n (voters), and m (candidates
per round) are not parameters here -- they are fixed by the dataset, so
they are printed to the terminal rather than taken as input.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from rich.console import Console

from ..encoding.encoding import save_decisions_json, save_profile_jsonl
from ..voting_rules.serial_dictator import SerialDictator
from .tsoi_parser import parse_tsoi_directory

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "experiments"

console = Console()


def run_natural_experiment(data_dir: Path | str) -> Path | None:
    """Load a temporal voting instance from data_dir (a directory of
    .tsoi files, one round per file), run the serial dictator rule on it,
    and save the approval profile and decision sequence to
    experiments/<data_dir name>/run_<n>/. Returns that run directory.

    T, n, and m are determined by the dataset rather than passed in; they
    are printed to the terminal before the rule is run.

    Errors are caught and reported as human-readable messages on stderr
    rather than raised; None is returned if the experiment could not be
    run or saved.
    """
    try:
        instance = parse_tsoi_directory(data_dir)
    except Exception as e:
        print(f"Error loading real data from '{data_dir}': {e}", file=sys.stderr)
        return None

    if not instance:
        print(f"Error: no approval profiles loaded from '{data_dir}'", file=sys.stderr)
        return None

    voters = list(instance[0].voters)
    T = len(instance)
    n = len(voters)
    cand_counts = {len(profile.cands) for profile in instance}
    if len(cand_counts) == 1:
        m_description = str(list(cand_counts)[0])
    else:
        m_description = f"{min(cand_counts)}-{max(cand_counts)}"

    console.print(f"Dataset: [bold]{Path(data_dir).name}[/bold]")
    console.print(f"T (rounds) = {T}")
    console.print(f"n (voters) = {n}")
    console.print(f"m (candidates per round) = {m_description}")

    try:
        serial_dictator: SerialDictator[int, int] = SerialDictator(voters=voters)
        decisions = serial_dictator(instance)
    except (ValueError, ZeroDivisionError, KeyError) as e:
        print(f"Error running serial dictator: {e}", file=sys.stderr)
        return None

    dataset_dir = EXPERIMENTS_DIR / Path(data_dir).name
    run_dir = dataset_dir / f"run_{_next_run_index(dataset_dir)}"
    try:
        run_dir.mkdir(parents=True)
    except OSError as e:
        print(f"Error creating run directory '{run_dir}': {e}", file=sys.stderr)
        return None

    approvals_path = run_dir / "approvals.jsonl"
    decisions_path = run_dir / "decisions.json"
    save_profile_jsonl(instance, approvals_path)
    save_decisions_json(decisions, decisions_path)

    if not approvals_path.exists() or not decisions_path.exists():
        print(f"Error: experiment data was not fully saved to {run_dir}", file=sys.stderr)
        return None

    print(f"Saved experiment to {run_dir}")
    return run_dir


def _next_run_index(dataset_dir: Path) -> int:
    """The next unused run_<n> index in dataset_dir, so repeated
    invocations accumulate runs instead of overwriting them."""
    if not dataset_dir.is_dir():
        return 0
    existing = [
        p.name for p in dataset_dir.iterdir() if p.is_dir() and re.fullmatch(r"run_\d+", p.name)
    ]
    if not existing:
        return 0
    return max(int(name.removeprefix("run_")) for name in existing) + 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a serial dictator experiment on real temporal voting data."
    )
    parser.add_argument("data_dir", type=Path, help="directory of .tsoi files, one round per file")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_natural_experiment(args.data_dir)
