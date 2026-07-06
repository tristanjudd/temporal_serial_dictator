"""Run a single synthetic serial-dictator experiment and save its results.

Generates a temporal approval voting instance of T rounds, runs the serial
dictator rule on it, and saves the approval profile and decision sequence
to a timestamped directory under src/experiments/.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from ..encoding.encoding import save_decisions_json, save_profile_jsonl
from ..voting_rules.serial_dictator import SerialDictator
from .instances import generate_instance

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"


def run_synthetic_experiment(
    T: int,
    n: int = 20,
    m: int = 5,
    sigma: float = 0.2,
    voter_point_mode: str = "eucl2",
    cand_point_mode: str = "uniform_square",
    approval_threshold: float = 1.5,
) -> Path:
    """Generate a synthetic instance of T rounds, run the serial dictator
    rule on it, and save the approval profile and decision sequence to a
    timestamped directory under src/experiments/. Returns that directory."""
    instance = generate_instance(
        n=n,
        m=m,
        T=T,
        sigma=sigma,
        voter_point_mode=voter_point_mode,
        cand_point_mode=cand_point_mode,
        approval_threshold=approval_threshold,
    )

    serial_dictator: SerialDictator[int, int] = SerialDictator(voters=list(range(n)))
    decisions = serial_dictator(instance)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir = EXPERIMENTS_DIR / f"T{T}-n{n}-{timestamp}"
    output_dir.mkdir(parents=True)

    save_profile_jsonl(instance, output_dir / "approvals.jsonl")
    save_decisions_json(decisions, output_dir / "decisions.json")

    print(f"Saved experiment (T={T}, n={n}) to {output_dir}")
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a synthetic serial dictator experiment.")
    parser.add_argument("T", type=int, help="number of rounds")
    parser.add_argument("--n", type=int, default=20, help="number of voters")
    parser.add_argument("--m", type=int, default=5, help="number of alternatives per round")
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--voter-point-mode", default="eucl2")
    parser.add_argument("--cand-point-mode", default="uniform_square")
    parser.add_argument("--approval-threshold", type=float, default=1.5)
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
    )
