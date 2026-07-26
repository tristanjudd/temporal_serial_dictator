"""Run a synthetic-grid-style (n, T) sweep over every real (natural)
temporal voting dataset.

Unlike real_aaai_experiment.py (which downsizes voters to one fixed
SAMPLE_SIZE and uses every round of the dataset), this mirrors
synthetic_data_tools.aaai_experiment's full (n, T) grid: for each real
dataset, samples voters at every N_VALUES entry the dataset can support,
and for each of those n, truncates to every T value the synthetic grid
uses for that n (imported directly from aaai_experiment.py, so the two
grids can never drift apart).

Truncating to T rounds always takes a *contiguous* window of the
dataset's rounds (never a scattered sample -- e.g. rounds 40..59, not
1, 5, 66, 566), since PJR's temporal structure depends on rounds being
consecutive. For every (dataset, n, T) configuration, the first of its
num_experiments runs always uses the dataset's actual first T rounds
(round_start=0) and is marked "is_first_rounds": true in its
metadata.json -- an unambiguous, greppable flag for pulling out these
baseline runs later for EDA, independent of whether some other run
happens to also start at round 0 by chance. Every other run uses a
freshly random contiguous window (and, if the dataset has more voters
than n, a freshly random voter sample -- the same sampling
run_natural_experiment.downsize_approval_profiles already does).

Reuses real_aaai_experiment._run_single_experiment to actually run and
save each run (identical save format/metadata.json convention), but has
its own dataset-iteration, grid-construction, and manifest logic, since
the (n, T) grid shape doesn't fit real_aaai_experiment's one-config-per-
dataset assumptions.
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
from ..synthetic_data_tools.aaai_experiment import N_VALUES, T_DIVISORS
from ..synthetic_data_tools.profiles import ApprovalProfile
from ..voting_rules.serial_dictator import SerialDictator
from .real_aaai_experiment import _run_single_experiment
from .run_natural_experiment import (
    EXPERIMENTS_DIR,
    _next_run_index,
    downsize_approval_profiles,
    load_jsonl_dataset,
)

console = Console()


def _build_configs(original_n: int, original_T: int) -> list[tuple[int, int]]:
    """The (n, T) pairs to run for a dataset with original_n voters and
    original_T rounds: every N_VALUES entry the dataset has enough
    voters for, paired with every T value aaai_experiment.build_grid
    would use for that n, that the dataset has enough rounds for.

    A dataset with fewer voters than the largest N_VALUES tier (e.g. 11,
    against N_VALUES=[5, 10, 20]) would otherwise never be run at its own
    full voter count -- n=10 still leaves a voter unused, and n=20 isn't
    feasible at all. So if original_n falls below the top tier and isn't
    already one of N_VALUES, it's added as an extra n, with its own T
    values derived the same way as any other n.
    """
    grid_n_values = list(N_VALUES)
    if original_n < max(N_VALUES) and original_n not in grid_n_values:
        grid_n_values.append(original_n)

    configs = []
    for n in sorted(grid_n_values):
        if n > original_n:
            continue
        t_values: list[int] = []
        for divisor in T_DIVISORS:
            t = n // divisor
            if t > 1 and t not in t_values:
                t_values.append(t)
        for t in t_values:
            if t <= original_T:
                configs.append((n, t))
    return configs


def run_real_aaai_grid_experiment(num_experiments: int = 10) -> None:
    """Run the (n, T) grid across every dataset in DATASETS_DIR
    (real_data/json_datasets).

    A dataset that raises any error partway through is reported and
    skipped, rather than aborting the remaining datasets.
    """
    if not DATASETS_DIR.is_dir():
        print(f"Error: '{DATASETS_DIR}' is not a directory.", file=sys.stderr)
        return

    dataset_paths = sorted(p for p in DATASETS_DIR.iterdir() if p.is_file())
    if not dataset_paths:
        print(f"Error: no datasets found in '{DATASETS_DIR}'.", file=sys.stderr)
        return

    for dataset_path in dataset_paths:
        try:
            _run_dataset_grid(dataset_path, num_experiments)
        except Exception as e:
            print(f"Error running experiment for '{dataset_path.name}': {e}", file=sys.stderr)
            continue


def _run_dataset_grid(dataset_path: Path, num_experiments: int) -> None:
    """Run every feasible (n, T) configuration for a single dataset, then
    (re)build its manifest and summary from the run_*/metadata.json files
    actually on disk.
    """
    loaded = load_jsonl_dataset(dataset_path)
    if loaded is None:
        return
    metadata, instance = loaded

    original_T = metadata["T"]
    original_n = len(instance[0].voters)
    configs = _build_configs(original_n, original_T)

    if not configs:
        print(
            f"Skipping {dataset_path.name}: no (n, T) grid configuration fits "
            f"{original_n} voter(s) / {original_T} round(s).",
            file=sys.stderr,
        )
        return

    dataset_dir = EXPERIMENTS_DIR / dataset_path.name
    console.print(
        f"[bold]{dataset_path.name}[/bold]: {original_n} voter(s), {original_T} round(s) "
        f"-> {len(configs)} configuration(s)"
    )

    for n, t in configs:
        _run_config(
            dataset_path.name, instance, original_n, original_T, n, t, num_experiments, dataset_dir
        )

    _write_grid_manifest(dataset_dir)


def _run_config(
    dataset_name: str,
    instance: list[ApprovalProfile],
    original_n: int,
    original_T: int,
    n: int,
    t: int,
    num_experiments: int,
    dataset_dir: Path,
) -> None:
    """Run num_experiments runs for a single (n, T) configuration.

    Run 0 always uses round_start=0 (the dataset's actual first t
    rounds) and is marked is_first_rounds=True; every other run uses a
    freshly random contiguous t-round window. Voters are downsized (a
    fresh random sample each run) whenever n < original_n, matching
    real_aaai_experiment's existing sampling behavior exactly.
    """
    downsize = n < original_n
    rng = random.Random()
    serial_dictator: SerialDictator[int, int] | None = None

    console.print(f"  n={n} T={t}: {num_experiments} run(s)")

    for i in track(range(num_experiments), description=f"{dataset_name} n={n} T={t}..."):
        is_first_rounds = i == 0
        round_start = 0 if is_first_rounds else rng.randint(0, original_T - t)
        windowed_instance = instance[round_start : round_start + t]

        run_instance = (
            downsize_approval_profiles(windowed_instance, n, rng) if downsize else windowed_instance
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
            "dataset": dataset_name,
            "T": t,
            "n": n,
            "downsized": downsize,
            "original_n": original_n,
            "original_T": original_T,
            "round_start": round_start,
            "is_first_rounds": is_first_rounds,
        }
        _run_single_experiment(run_instance, serial_dictator, run_dir, run_metadata)


def _write_grid_manifest(dataset_dir: Path) -> None:
    """Rebuild experiment_manifest.jsonl and experiment_summary.log for
    dataset_dir from its run_*/metadata.json files, grouped by (n, T)
    configuration, so both always reflect what's actually on disk.
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

    config_runs: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for entry in manifest_entries:
        key = (entry.get("n"), entry.get("T"))
        config_runs.setdefault(key, []).append(entry)

    summary_lines = [
        f"Real-data grid experiment: {dataset_dir}",
        f"Total runs: {len(manifest_entries)}",
        f"Configurations: {len(config_runs)}",
        "",
    ]
    for (n, t), entries in sorted(config_runs.items()):
        first_rounds_entry = next((e for e in entries if e.get("is_first_rounds")), None)
        first_rounds_run = first_rounds_entry["run"] if first_rounds_entry else "?"
        run_names = [e["run"] for e in entries]
        summary_lines.append(
            f"n={n} T={t}: {len(entries)} run(s) ({run_names[0]}..{run_names[-1]}), "
            f"first-rounds run: {first_rounds_run}"
        )
    summary_text = "\n".join(summary_lines)

    summary_path = dataset_dir / "experiment_summary.log"
    try:
        summary_path.write_text(summary_text + "\n")
    except OSError as e:
        print(f"Error writing summary log to '{summary_path}': {e}", file=sys.stderr)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the synthetic-grid-style (n, T) sweep across every real dataset."
    )
    parser.add_argument(
        "--num-experiments",
        type=int,
        default=10,
        help="number of runs per (dataset, n, T) configuration",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_real_aaai_grid_experiment(num_experiments=args.num_experiments)
