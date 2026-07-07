"""Verify Proportional Justified Representation (PJR) for saved experiments.

Loads each run's approval profile and decision sequence from a saved
experiment directory (as produced by run_synthetic_experiment.py), checks
PJR for every subset of voters, and reports violations.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import track

from ..encoding.decoding import load_decisions_json, load_profile_jsonl
from ..synthetic_data.profiles import ApprovalProfile

console = Console()


def find_pjr_violations(
    instance: list[ApprovalProfile], decisions: list[Any]
) -> list[dict[str, Any]]:
    """Check PJR for every non-empty subset of voters in a single run.

    For a voter group, PJR requires: if the group agrees on a common
    candidate in `agreement` rounds, then some voter in the group must be
    satisfied (approve the round's winner) in at least
    floor(agreement * |group| / n) rounds.

    Returns the list of violating groups, each as
    {"voters", "agreement", "bound", "satisfaction"}.
    """
    if not instance:
        return []

    voters = list(instance[0].voters)
    n = len(voters)

    all_groups = list(
        itertools.chain.from_iterable(
            itertools.combinations(voters, size) for size in range(1, n + 1)
        )
    )

    violations = []
    for group in track(all_groups, description="Verifying PJR..."):
        agreement = 0
        satisfaction = 0
        for profile, winner in zip(instance, decisions, strict=True):
            approval_sets = [set(profile.approval_sets[voter]) for voter in group]
            if set.intersection(*approval_sets):
                agreement += 1
            if any(winner in profile.approval_sets[voter] for voter in group):
                satisfaction += 1

        bound = (agreement * len(group)) // n
        if satisfaction < bound:
            violations.append(
                {
                    "voters": list(group),
                    "agreement": agreement,
                    "bound": bound,
                    "satisfaction": satisfaction,
                }
            )
    return violations


def verify_run(run_dir: Path) -> dict[str, Any] | None:
    """Load a run's approval profile and decision sequence, check PJR for
    every voter subset, and save the violations to run_dir/violations.jsonl.

    Returns {"num_violations", "worst"} for this run ("worst" is the
    violation with the largest bound-satisfaction gap, or None if there
    were no violations), or None if the run's data could not be loaded or
    the violations could not be saved.
    """
    instance = load_profile_jsonl(run_dir / "approvals.jsonl")
    decisions = load_decisions_json(run_dir / "decisions.json")
    if instance is None or decisions is None:
        print(f"Skipping {run_dir}: could not load approvals/decisions.", file=sys.stderr)
        return None

    violations = find_pjr_violations(instance, decisions)

    violations_path = run_dir / "violations.jsonl"
    try:
        with violations_path.open("w") as f:
            for violation in violations:
                f.write(json.dumps(violation) + "\n")
    except OSError as e:
        print(f"Error writing violations to '{violations_path}': {e}", file=sys.stderr)
        return None

    worst = max(violations, key=lambda v: v["bound"] - v["satisfaction"], default=None)

    if worst is None:
        console.print(f"{run_dir.name}: [bold green]PJR satisfied[/bold green]")
    else:
        gap = worst["bound"] - worst["satisfaction"]
        console.print(
            f"{run_dir.name}: [bold red]PJR violated[/bold red] "
            f"({len(violations)} violation(s), worst: group {worst['voters']} "
            f"bound={worst['bound']} satisfaction={worst['satisfaction']} gap={gap})"
        )

    return {"num_violations": len(violations), "worst": worst}


def verify_experiment(experiment_dir: Path) -> None:
    """Verify PJR for every run_* subdirectory of experiment_dir.

    Saves a violations.jsonl into each run subdirectory, and a summary
    log (whether PJR is satisfied, how many violations, and the worst
    one) into experiment_dir itself. The same summary is printed to
    stdout.
    """
    if not experiment_dir.is_dir():
        print(f"Error: '{experiment_dir}' is not a directory.", file=sys.stderr)
        return

    run_dirs = sorted(
        (p for p in experiment_dir.iterdir() if p.is_dir() and p.name.startswith("run_")),
        key=lambda p: int(p.name.removeprefix("run_")),
    )
    if not run_dirs:
        print(f"Error: no run_* subdirectories found in '{experiment_dir}'.", file=sys.stderr)
        return

    num_runs_violating = 0
    total_violations = 0
    worst_overall: tuple[str, int, dict[str, Any]] | None = None

    for run_dir in run_dirs:
        summary = verify_run(run_dir)
        if summary is None:
            continue

        total_violations += summary["num_violations"]
        if summary["num_violations"] > 0:
            num_runs_violating += 1

        worst = summary["worst"]
        if worst is not None:
            gap = worst["bound"] - worst["satisfaction"]
            if worst_overall is None or gap > worst_overall[1]:
                worst_overall = (run_dir.name, gap, worst)

    satisfied = num_runs_violating == 0
    summary_lines = [
        f"PJR verification for {experiment_dir}",
        f"Runs checked: {len(run_dirs)}",
        f"PJR satisfied: {satisfied}",
        f"Runs violating PJR: {num_runs_violating}",
        f"Total violating groups: {total_violations}",
    ]
    if worst_overall is not None:
        run_name, gap, worst = worst_overall
        summary_lines.append(
            f"Worst violation: run {run_name}, group {worst['voters']}, "
            f"bound={worst['bound']}, satisfaction={worst['satisfaction']}, gap={gap}"
        )
    summary_text = "\n".join(summary_lines)

    log_path = experiment_dir / "pjr_summary.log"
    try:
        log_path.write_text(summary_text + "\n")
    except OSError as e:
        print(f"Error writing summary log to '{log_path}': {e}", file=sys.stderr)

    status_style = "bold green" if satisfied else "bold red"
    status_text = "SATISFIED" if satisfied else "VIOLATED"
    console.print(f"[bold]PJR verification for[/bold] {experiment_dir}")
    console.print(f"Runs checked: {len(run_dirs)}")
    console.print(f"PJR status: [{status_style}]{status_text}[/{status_style}]")
    console.print(f"Runs violating PJR: {num_runs_violating}")
    console.print(f"Total violating groups: {total_violations}")
    if worst_overall is not None:
        run_name, gap, worst = worst_overall
        console.print(
            f"Worst violation: run [bold]{run_name}[/bold], group {worst['voters']}, "
            f"bound={worst['bound']}, satisfaction={worst['satisfaction']}, gap={gap}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify PJR for a saved experiment.")
    parser.add_argument(
        "experiment_dir", type=Path, help="path to an experiment directory (contains run_* dirs)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    verify_experiment(args.experiment_dir)
