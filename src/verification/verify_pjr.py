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
from ..synthetic_data_tools.profiles import ApprovalProfile

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


def _load_metadata(run_dir: Path) -> dict[str, Any] | None:
    """Load run_dir/metadata.json if it exists.

    Returns None if there is no metadata file, or if it could not be
    parsed -- runs without metadata (e.g. predating metadata.json) are
    still verified normally, just without parameter info in the logs.
    """
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: could not read metadata from '{metadata_path}': {e}", file=sys.stderr)
        return None


def _format_metadata_plain(metadata: dict[str, Any] | None) -> str:
    """Format a run's metadata as "key=value, key=value, ...". Empty
    string if there is no metadata to show.
    """
    if not metadata:
        return ""
    return ", ".join(f"{key}={value}" for key, value in metadata.items())


def _format_metadata(metadata: dict[str, Any] | None) -> str:
    """Format a run's metadata as a trailing string for log/console
    lines, e.g. " (n=5, m=10, T=5, approval_threshold=1.5)". Empty
    string if there is no metadata to show.
    """
    plain = _format_metadata_plain(metadata)
    return f" ({plain})" if plain else ""


def verify_run(run_dir: Path) -> dict[str, Any] | None:
    """Load a run's approval profile and decision sequence, check PJR for
    every voter subset, and save the violations to run_dir/violations.jsonl.

    Returns {"num_violations", "worst", "metadata"} for this run ("worst"
    is the violation with the largest bound-satisfaction gap, or None if
    there were no violations; "metadata" is the run's metadata.json
    contents, or None if it doesn't exist), or None if the run's data
    could not be loaded or the violations could not be saved.
    """
    instance = load_profile_jsonl(run_dir / "approvals.jsonl")
    decisions = load_decisions_json(run_dir / "decisions.json")
    if instance is None or decisions is None:
        print(f"Skipping {run_dir}: could not load approvals/decisions.", file=sys.stderr)
        return None

    metadata = _load_metadata(run_dir)
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
    metadata_str = _format_metadata(metadata)

    if worst is None:
        console.print(f"{run_dir.name}: [bold green]PJR satisfied[/bold green]{metadata_str}")
    else:
        gap = worst["bound"] - worst["satisfaction"]
        console.print(
            f"{run_dir.name}: [bold red]PJR violated[/bold red] "
            f"({len(violations)} violation(s), worst: group {worst['voters']} "
            f"bound={worst['bound']} satisfaction={worst['satisfaction']} gap={gap})"
            f"{metadata_str}"
        )

    return {"num_violations": len(violations), "worst": worst, "metadata": metadata}


def verify_experiment(experiment_dir: Path) -> None:
    """Verify PJR for every run_* subdirectory of experiment_dir.

    Saves a violations.jsonl into each run subdirectory, and a summary
    log (whether PJR is satisfied, how many violations, the worst one,
    and -- for runs with metadata.json -- a breakdown of violations by
    configuration, for configurations with at least one violation) into
    experiment_dir itself. The same summary is printed to stdout.
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
    worst_overall: tuple[str, int, dict[str, Any], dict[str, Any] | None] | None = None
    # keyed by a canonical (sorted-key) json rendering of a run's metadata,
    # so runs sharing the same configuration accumulate into the same entry.
    config_stats: dict[str, dict[str, Any]] = {}

    for run_dir in run_dirs:
        summary = verify_run(run_dir)
        if summary is None:
            continue

        total_violations += summary["num_violations"]
        if summary["num_violations"] > 0:
            num_runs_violating += 1

        worst = summary["worst"]
        metadata = summary["metadata"]
        if worst is not None:
            gap = worst["bound"] - worst["satisfaction"]
            if worst_overall is None or gap > worst_overall[1]:
                worst_overall = (run_dir.name, gap, worst, metadata)

            if metadata:
                config_key = json.dumps(metadata, sort_keys=True)
                stats = config_stats.setdefault(
                    config_key,
                    {"metadata": metadata, "num_violations": 0, "worst_gap": -1, "worst": None},
                )
                stats["num_violations"] += summary["num_violations"]
                if gap > stats["worst_gap"]:
                    stats["worst_gap"] = gap
                    stats["worst"] = worst
                    stats["worst_run"] = run_dir.name

    satisfied = num_runs_violating == 0
    summary_lines = [
        f"PJR verification for {experiment_dir}",
        f"Runs checked: {len(run_dirs)}",
        f"PJR satisfied: {satisfied}",
        f"Runs violating PJR: {num_runs_violating}",
        f"Total violating groups: {total_violations}",
    ]
    if worst_overall is not None:
        run_name, gap, worst, metadata = worst_overall
        summary_lines.append(
            f"Worst violation: run {run_name}, group {worst['voters']}, "
            f"bound={worst['bound']}, satisfaction={worst['satisfaction']}, gap={gap}"
            f"{_format_metadata(metadata)}"
        )
    if config_stats:
        summary_lines.append("")
        summary_lines.append("Violations by configuration:")
        for stats in sorted(config_stats.values(), key=lambda s: s["num_violations"], reverse=True):
            worst = stats["worst"]
            summary_lines.append(
                f"  {_format_metadata_plain(stats['metadata'])}: "
                f"{stats['num_violations']} violation(s), "
                f"worst: run {stats['worst_run']}, group {worst['voters']}, "
                f"bound={worst['bound']}, satisfaction={worst['satisfaction']}, "
                f"gap={stats['worst_gap']}"
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
        run_name, gap, worst, metadata = worst_overall
        console.print(
            f"Worst violation: run [bold]{run_name}[/bold], group {worst['voters']}, "
            f"bound={worst['bound']}, satisfaction={worst['satisfaction']}, gap={gap}"
            f"{_format_metadata(metadata)}"
        )
    if config_stats:
        console.print("[bold]Violations by configuration:[/bold]")
        for stats in sorted(config_stats.values(), key=lambda s: s["num_violations"], reverse=True):
            worst = stats["worst"]
            console.print(
                f"  {_format_metadata_plain(stats['metadata'])}: "
                f"{stats['num_violations']} violation(s), "
                f"worst: run {stats['worst_run']}, group {worst['voters']}, "
                f"bound={worst['bound']}, satisfaction={worst['satisfaction']}, "
                f"gap={stats['worst_gap']}"
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
