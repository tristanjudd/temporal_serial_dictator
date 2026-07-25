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
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.console import Console

from .run_synthetic_experiment import EXPERIMENTS_DIR, SEED_SPACE, run_synthetic_experiment

console = Console()

# Matches Lackner's own random.seed(31415) in experiments_aaai.py -- the
# default here makes the whole grid reproducible out of the box, not just
# when a seed happens to be explicitly requested.
DEFAULT_SEED = 31415

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

    T is derived per n from T_DIVISORS as n // divisor; duplicate T
    values for a given n (multiple divisors flooring to the same T)
    collapse to a single configuration rather than being run twice. T=1
    is excluded entirely: with only one round, the only group that can
    qualify for representation is everyone (N itself), and any candidate
    approved by anyone trivially satisfies it -- so T=1 instances are
    trivially PJR-satisfying and not worth including in the grid.
    """
    configs = []
    for n in N_VALUES:
        t_values: list[int] = []
        for divisor in T_DIVISORS:
            t = n // divisor
            if t > 1 and t not in t_values:
                t_values.append(t)
        for m, t, threshold in itertools.product(M_VALUES, t_values, THRESHOLDS):
            configs.append(Config(n=n, m=m, T=t, approval_threshold=threshold))
    return configs


def run_aaai_experiment(num_experiments: int = 10, seed: int = DEFAULT_SEED) -> Path | None:
    """Run the full (n, m, T, threshold) grid, num_experiments runs per
    configuration, all as flat run_* subdirectories under a single parent
    directory under experiments/.

    seed makes the entire grid reproducible: a single random.Random(seed)
    derives one sub-seed per configuration (in grid order), which is
    passed to that configuration's run_synthetic_experiment call. The
    same seed always reproduces the exact same grid, including every
    run's instance and every SerialDictator's sequence of choices.

    Saves experiment_manifest.jsonl (one line per run, giving its
    configuration and seed) and experiment_summary.log (a human-readable
    per-configuration summary) into the parent directory. Returns the
    parent directory, or None if no runs succeeded.
    """
    configs = build_grid()
    seed_rng = random.Random(seed)
    config_seeds = [seed_rng.randrange(SEED_SPACE) for _ in range(len(configs))]

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
        f"Seed: {seed}",
        f"Configurations: {len(configs)}",
        f"Runs per configuration: {num_experiments}",
        f"Total runs: {total_runs}",
        "",
    ]

    run_offset = 0
    for config_id, config in enumerate(configs):
        config_seed = config_seeds[config_id]
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
            seed=config_seed,
        )

        num_succeeded = sum(run_dir is not None for run_dir in run_dirs)
        summary_lines.append(
            f"config {config_id}: n={config.n} m={config.m} T={config.T} "
            f"threshold={config.approval_threshold} seed={config_seed} -> "
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
                    "seed": config_seed,
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
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"seed for reproducibility (default: {DEFAULT_SEED})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_aaai_experiment(num_experiments=args.num_experiments, seed=args.seed)
