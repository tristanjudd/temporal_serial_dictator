"""Synthetic experiment grid for the paper.

Sweeps n (voters), m (candidates/round), T (rounds), and approval
threshold, running num_experiments serial dictator runs per configuration.
All runs are saved as flat run_<i>/ subdirectories under a single parent
directory, so the whole grid can be handed directly to the existing PJR
verification tools (verify-pjr / multi-verify-pjr) without modification.

Since verification results (violations.jsonl) are per-run and don't carry
configuration info, this script also saves experiment_manifest.jsonl (one
line per run, giving its n/m/T/threshold) so violations can be joined back
to the configuration that produced them.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.console import Console

from .run_synthetic_experiment import EXPERIMENTS_DIR, run_synthetic_experiment

console = Console()

N_VALUES = [5, 10, 20]
T_DIVISORS = [1, 2, 4, 8]
M_VALUES = [10, 20, 50]
THRESHOLDS = [1.5, 2.0, 2.5, 3.0]

SIGMA = 0.2
VOTER_POINT_MODE = "eucl2"
CAND_POINT_MODE = "uniform_square"


@dataclass
class Config:
    n: int
    m: int
    T: int
    approval_threshold: float


def build_grid() -> list[Config]:
    """Build the (n, m, T, threshold) configuration grid.

    T is derived per n from T_DIVISORS as n // divisor (minimum 1); for
    small n, several divisors can floor to the same T (e.g. n=5 gives T=1
    for both divisor 4 and 8), so duplicate T values per n collapse to a
    single configuration rather than being run twice.
    """
    configs = []
    for n in N_VALUES:
        t_values: list[int] = []
        for divisor in T_DIVISORS:
            t = max(1, n // divisor)
            if t not in t_values:
                t_values.append(t)
        for m, t, threshold in itertools.product(M_VALUES, t_values, THRESHOLDS):
            configs.append(Config(n=n, m=m, T=t, approval_threshold=threshold))
    return configs


def run_aaai_experiment(num_experiments: int = 10) -> Path | None:
    """Run the full (n, m, T, threshold) grid, num_experiments runs per
    configuration, all as flat run_* subdirectories under a single parent
    directory under experiments/.

    Saves experiment_manifest.jsonl (one line per run, giving its
    configuration) and experiment_summary.log (a human-readable
    per-configuration summary) into the parent directory. Returns the
    parent directory, or None if no runs succeeded.
    """
    configs = build_grid()
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    parent_dir = EXPERIMENTS_DIR / f"aaai-grid-{timestamp}"

    total_runs = len(configs) * num_experiments
    console.print(
        f"[bold]Running {len(configs)} configurations x {num_experiments} "
        f"experiment(s) = {total_runs} runs into[/bold] {parent_dir}"
    )

    manifest_entries = []
    summary_lines = [
        f"AAAI grid experiment: {parent_dir}",
        f"Configurations: {len(configs)}",
        f"Runs per configuration: {num_experiments}",
        f"Total runs: {total_runs}",
        "",
    ]

    run_offset = 0
    for config_id, config in enumerate(configs):
        console.print(
            f"[bold]Configuration {config_id + 1}/{len(configs)}[/bold]: "
            f"n={config.n} m={config.m} T={config.T} threshold={config.approval_threshold}"
        )
        run_dirs = run_synthetic_experiment(
            T=config.T,
            n=config.n,
            m=config.m,
            sigma=SIGMA,
            voter_point_mode=VOTER_POINT_MODE,
            cand_point_mode=CAND_POINT_MODE,
            approval_threshold=config.approval_threshold,
            num_experiments=num_experiments,
            experiment_dir=parent_dir,
            run_offset=run_offset,
        )

        num_succeeded = sum(run_dir is not None for run_dir in run_dirs)
        summary_lines.append(
            f"config {config_id}: n={config.n} m={config.m} T={config.T} "
            f"threshold={config.approval_threshold} -> "
            f"run_{run_offset}..run_{run_offset + num_experiments - 1} "
            f"({num_succeeded}/{num_experiments} succeeded)"
        )

        for run_dir in run_dirs:
            if run_dir is None:
                continue
            manifest_entries.append(
                {
                    "run": run_dir.name,
                    "config_id": config_id,
                    "n": config.n,
                    "m": config.m,
                    "T": config.T,
                    "sigma": SIGMA,
                    "voter_point_mode": VOTER_POINT_MODE,
                    "cand_point_mode": CAND_POINT_MODE,
                    "approval_threshold": config.approval_threshold,
                }
            )

        run_offset += num_experiments

    if not manifest_entries:
        print("Error: no runs succeeded across the entire grid.", file=sys.stderr)
        return None

    manifest_path = parent_dir / "experiment_manifest.jsonl"
    try:
        with manifest_path.open("w") as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry) + "\n")
    except OSError as e:
        print(f"Error writing manifest to '{manifest_path}': {e}", file=sys.stderr)
        return None

    summary_path = parent_dir / "experiment_summary.log"
    try:
        summary_path.write_text("\n".join(summary_lines) + "\n")
    except OSError as e:
        print(f"Error writing summary log to '{summary_path}': {e}", file=sys.stderr)

    console.print(
        f"[bold green]Saved {len(manifest_entries)}/{total_runs} runs across "
        f"{len(configs)} configurations to[/bold green] {parent_dir}"
    )
    return parent_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper's synthetic experiment grid "
        "(sweep over n, m, T, approval threshold)."
    )
    parser.add_argument(
        "--num-experiments", type=int, default=10, help="number of runs per configuration"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_aaai_experiment(num_experiments=args.num_experiments)
