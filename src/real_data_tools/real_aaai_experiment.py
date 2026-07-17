"""Run serial dictator experiments across every real (natural) temporal
voting dataset, for the paper.

Mirrors synthetic_data_tools.aaai_experiment's design and logging, but
sources instances from real (converted) datasets instead of synthetic
generation. It calls run_natural_experiment's building blocks (dataset
loading, voter downsizing, run-index bookkeeping) directly rather than
run_natural_experiment() itself, since saving per-run metadata.json (to
match the synthetic experiment's logging) requires hooking into the
save step that run_natural_experiment() does not expose.

Datasets with more voters than SAMPLE_SIZE get num_experiments
independent runs, each on a fresh random sample of SAMPLE_SIZE voters
(see run_natural_experiment.downsize_approval_profiles); datasets with
SAMPLE_SIZE voters or fewer get exactly one run on the full voter set.
Each dataset gets its own directory under experiments/ (named after the
dataset), holding all of that dataset's runs plus an
experiment_manifest.jsonl and experiment_summary.log, matching the
artifacts synthetic_data_tools.aaai_experiment produces.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import track

from ..data_transformation.convert_lackner_datasets import OUTPUT_DIR as DATASETS_DIR
from ..encoding.encoding import save_decisions_json, save_profile_jsonl
from ..synthetic_data_tools.profiles import ApprovalProfile
from ..voting_rules.serial_dictator import SerialDictator
from .run_natural_experiment import (
    EXPERIMENTS_DIR,
    _next_run_index,
    downsize_approval_profiles,
    load_jsonl_dataset,
)

console = Console()

SAMPLE_SIZE = 20


def run_real_aaai_experiment(num_experiments: int = 10) -> None:
    """Run serial dictator experiments across every dataset in
    DATASETS_DIR (real_data/json_datasets).

    Datasets with more than SAMPLE_SIZE voters get num_experiments runs,
    each on a fresh random sample of SAMPLE_SIZE voters; datasets with
    SAMPLE_SIZE voters or fewer get exactly one run on their full voter
    set.
    """
    if not DATASETS_DIR.is_dir():
        print(f"Error: '{DATASETS_DIR}' is not a directory.", file=sys.stderr)
        return

    dataset_paths = sorted(p for p in DATASETS_DIR.iterdir() if p.is_file())
    if not dataset_paths:
        print(f"Error: no datasets found in '{DATASETS_DIR}'.", file=sys.stderr)
        return

    for dataset_path in dataset_paths:
        _run_dataset(dataset_path, num_experiments)


def _run_dataset(dataset_path: Path, num_experiments: int) -> None:
    """Run all experiments for a single dataset, saving each run's
    metadata.json alongside its approval profile and decision sequence,
    then (re)build the dataset's manifest and summary from the
    run_*/metadata.json files actually on disk.
    """
    loaded = load_jsonl_dataset(dataset_path)
    if loaded is None:
        return
    metadata, instance = loaded

    T = metadata["T"]
    original_n = len(instance[0].voters)
    downsize = original_n > SAMPLE_SIZE
    n = SAMPLE_SIZE if downsize else original_n
    runs_to_do = num_experiments if downsize else 1

    console.print(
        f"[bold]{dataset_path.name}[/bold]: T={T}, {original_n} voter(s) -> "
        + (f"sampling {n} voter(s), {runs_to_do} run(s)" if downsize else "1 run (full voter set)")
    )

    dataset_dir = EXPERIMENTS_DIR / dataset_path.name
    sampling_random_state = random.Random()
    serial_dictator: SerialDictator[int, int] | None = None

    for _ in track(range(runs_to_do), description=f"Running {dataset_path.name}..."):
        run_instance = (
            downsize_approval_profiles(instance, SAMPLE_SIZE, sampling_random_state)
            if downsize
            else instance
        )
        voters = list(run_instance[0].voters)

        if serial_dictator is None:
            serial_dictator = SerialDictator(voters=voters)
        else:
            serial_dictator.voters = voters
            serial_dictator.permutation = list(voters)
            serial_dictator.reset()

        run_dir = dataset_dir / f"run_{_next_run_index(dataset_dir)}"
        run_metadata = {
            "dataset": dataset_path.name,
            "T": T,
            "n": n,
            "downsized": downsize,
            "original_n": original_n,
        }
        _run_single_experiment(run_instance, serial_dictator, run_dir, run_metadata)

    _write_dataset_manifest(dataset_dir)


def _run_single_experiment(
    run_instance: list[ApprovalProfile],
    serial_dictator: SerialDictator[int, int],
    run_dir: Path,
    metadata: dict[str, Any],
) -> Path | None:
    """Run serial_dictator (already reset, if reused) on run_instance,
    and save the approval profile, decision sequence, and metadata to
    run_dir. Returns run_dir.

    Errors are caught and reported as human-readable messages on stderr
    rather than raised; None is returned if the run could not be
    completed or saved.
    """
    try:
        decisions = serial_dictator(run_instance)
    except (ValueError, ZeroDivisionError, KeyError) as e:
        print(f"Error running serial dictator: {e}", file=sys.stderr)
        return None

    try:
        run_dir.mkdir(parents=True)
    except OSError as e:
        print(f"Error creating run directory '{run_dir}': {e}", file=sys.stderr)
        return None

    approvals_path = run_dir / "approvals.jsonl"
    decisions_path = run_dir / "decisions.json"
    metadata_path = run_dir / "metadata.json"
    save_profile_jsonl(run_instance, approvals_path)
    save_decisions_json(decisions, decisions_path)
    try:
        metadata_path.write_text(json.dumps(metadata))
    except OSError as e:
        print(f"Error writing metadata to '{metadata_path}': {e}", file=sys.stderr)
        return None

    if not approvals_path.exists() or not decisions_path.exists() or not metadata_path.exists():
        print(f"Error: experiment data was not fully saved to {run_dir}", file=sys.stderr)
        return None

    return run_dir


def _write_dataset_manifest(dataset_dir: Path) -> None:
    """Rebuild experiment_manifest.jsonl and experiment_summary.log for
    dataset_dir from its run_*/metadata.json files, so both always
    reflect what's actually on disk (including runs from earlier calls)
    rather than just the runs added in this call.
    """
    if not dataset_dir.is_dir():
        return

    run_dirs = sorted(
        (p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("run_")),
        key=lambda p: int(p.name.removeprefix("run_")),
    )

    manifest_entries = []
    for run_dir in run_dirs:
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            run_metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: could not read metadata from '{metadata_path}': {e}", file=sys.stderr)
            continue
        manifest_entries.append({"run": run_dir.name, **run_metadata})

    manifest_path = dataset_dir / "experiment_manifest.jsonl"
    try:
        with manifest_path.open("w") as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry) + "\n")
    except OSError as e:
        print(f"Error writing manifest to '{manifest_path}': {e}", file=sys.stderr)
        return

    summary_lines = [
        f"Real-data experiment: {dataset_dir}",
        f"Total runs: {len(manifest_entries)}",
    ]
    if manifest_entries:
        first = manifest_entries[0]
        num_downsized = sum(1 for entry in manifest_entries if entry.get("downsized"))
        summary_lines.extend(
            [
                f"Dataset: {first.get('dataset')}",
                f"T (rounds): {first.get('T')}",
                f"Original voters: {first.get('original_n')}",
                f"Downsized runs: {num_downsized}/{len(manifest_entries)}",
            ]
        )
    summary_text = "\n".join(summary_lines)

    summary_path = dataset_dir / "experiment_summary.log"
    try:
        summary_path.write_text(summary_text + "\n")
    except OSError as e:
        print(f"Error writing summary log to '{summary_path}': {e}", file=sys.stderr)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper's real-data experiment grid "
        f"(all datasets, downsized to {SAMPLE_SIZE} voters where necessary)."
    )
    parser.add_argument(
        "--num-experiments",
        type=int,
        default=10,
        help="number of runs per dataset that needs downsizing",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_real_aaai_experiment(num_experiments=args.num_experiments)
