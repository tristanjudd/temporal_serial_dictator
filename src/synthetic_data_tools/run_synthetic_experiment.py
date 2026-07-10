"""Run synthetic serial-dictator experiments and save their results.

Generates temporal approval voting instances of T rounds, runs the serial
dictator rule on each, and saves the approval profile and decision sequence
under experiments/<EXPERIMENT>/<RUN>/, where <EXPERIMENT> identifies the
whole batch and <RUN> is one individual run within it.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.progress import track

from ..encoding.encoding import save_decisions_json, save_profile_jsonl
from ..voting_rules.serial_dictator import SerialDictator
from .instances import generate_instance

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "experiments"


def run_synthetic_experiment(
    T: int,
    n: int = 20,
    m: int = 5,
    sigma: float = 0.2,
    voter_point_mode: str = "eucl2",
    cand_point_mode: str = "uniform_square",
    approval_threshold: float = 1.5,
    num_experiments: int = 1,
) -> list[Path | None]:
    """Run num_experiments independent synthetic serial dictator
    experiments, each with a freshly generated instance of T rounds.

    Creates one experiment directory under experiments/, and within it one
    run_<i>/ subdirectory per experiment, each holding that run's approval
    profile and decision sequence.

    Displays a progress bar for the experiments as they run, and prints a
    summary with the total elapsed time once all of them are done.

    Returns the list of run directories, one per experiment, in order; an
    entry is None if that particular experiment failed.
    """
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    experiment_dir = EXPERIMENTS_DIR / f"T{T}-n{n}-{timestamp}"

    # One SerialDictator, reused (and reset) across every run in the batch,
    # so all runs share the same voter ordering and only the synthetic data
    # varies from run to run.
    serial_dictator: SerialDictator[int, int] = SerialDictator(voters=list(range(n)))

    start_time = time.perf_counter()

    results = []
    for i in track(range(num_experiments), description="Running experiments..."):
        results.append(
            _run_single_experiment(
                T=T,
                n=n,
                m=m,
                sigma=sigma,
                voter_point_mode=voter_point_mode,
                cand_point_mode=cand_point_mode,
                approval_threshold=approval_threshold,
                serial_dictator=serial_dictator,
                run_dir=experiment_dir / f"run_{i}",
                announce=False,
            )
        )

    elapsed = time.perf_counter() - start_time
    num_succeeded = sum(result is not None for result in results)
    print(
        f"Completed {num_experiments} experiment(s) in {elapsed:.2f}s: "
        f"{num_succeeded} succeeded, {num_experiments - num_succeeded} failed."
    )
    return results


def _run_single_experiment(
    T: int,
    n: int,
    m: int,
    sigma: float,
    voter_point_mode: str,
    cand_point_mode: str,
    approval_threshold: float,
    serial_dictator: SerialDictator[int, int],
    run_dir: Path,
    announce: bool = True,
) -> Path | None:
    """Generate a synthetic instance of T rounds, run serial_dictator on
    it, and save the approval profile and decision sequence to run_dir.
    Returns run_dir.

    serial_dictator is reset (but not re-permuted) before running, so
    repeated calls sharing the same SerialDictator instance all start
    their rounds at permutation[0] again.

    Errors are caught and reported as human-readable messages on stderr
    rather than raised; None is returned if the experiment could not be
    run or saved.
    """
    try:
        instance = generate_instance(
            n=n,
            m=m,
            T=T,
            sigma=sigma,
            voter_point_mode=voter_point_mode,
            cand_point_mode=cand_point_mode,
            approval_threshold=approval_threshold,
        )
    except ValueError as e:
        print(f"Error generating synthetic instance: {e}", file=sys.stderr)
        return None

    try:
        serial_dictator.reset()
        decisions = serial_dictator(instance)
    except (ValueError, ZeroDivisionError) as e:
        print(f"Error running serial dictator: {e}", file=sys.stderr)
        return None

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

    if announce:
        print(f"Saved experiment (T={T}, n={n}) to {run_dir}")
    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic serial dictator experiments.")
    parser.add_argument("T", type=int, help="number of rounds")
    parser.add_argument("--n", type=int, default=20, help="number of voters")
    parser.add_argument("--m", type=int, default=5, help="number of alternatives per round")
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--voter-point-mode", default="eucl2")
    parser.add_argument("--cand-point-mode", default="uniform_square")
    parser.add_argument("--approval-threshold", type=float, default=1.5)
    parser.add_argument(
        "--num-experiments", type=int, default=1, help="number of experiments to run"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_synthetic_experiment(
        T=args.T,
        n=args.n,
        m=args.m,
        sigma=args.sigma,
        voter_point_mode=args.voter_point_mode,
        cand_point_mode=args.cand_point_mode,
        approval_threshold=args.approval_threshold,
        num_experiments=args.num_experiments,
    )
