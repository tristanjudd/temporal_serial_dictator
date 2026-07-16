"""Multiprocessing-parallelized PJR verification for saved experiments.

Same PJR check as verify_pjr.py, but the O(2^n) voter-subset enumeration
for a single run is split across a process pool (true multiprocessing,
not threads -- each worker is a separate OS process, so this actually
uses multiple CPU cores) instead of being checked one group at a time in
a single process. Parallelization is over groups within a single run's
verification, not over runs or experiments, since a single group's check
is already fast and linear -- the bottleneck is the sheer number of
groups.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import track

from ..encoding.decoding import load_decisions_json, load_profile_jsonl
from ..synthetic_data_tools.profiles import ApprovalProfile

console = Console()

# Set once per worker process by _init_worker, then reused for every
# chunk that worker processes -- so the (read-only) instance/decisions
# are only ever pickled once per worker, not once per chunk.
_worker_instance: list[ApprovalProfile] | None = None
_worker_decisions: list[Any] | None = None
_worker_n: int = 0


def _init_worker(instance: list[ApprovalProfile], decisions: list[Any]) -> None:
    """Pool initializer: runs once when each worker process starts."""
    global _worker_instance, _worker_decisions, _worker_n
    _worker_instance = instance
    _worker_decisions = decisions
    _worker_n = len(instance[0].voters) if instance else 0


def _check_groups_chunk(groups: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
    """Check a chunk of voter groups for PJR violations. Runs inside a
    worker process, using the instance/decisions/n stashed by
    _init_worker.
    """
    assert _worker_instance is not None and _worker_decisions is not None

    violations = []
    for group in groups:
        agreement = 0
        satisfaction = 0
        for profile, winner in zip(_worker_instance, _worker_decisions, strict=True):
            approval_sets = [set(profile.approval_sets[voter]) for voter in group]
            if set.intersection(*approval_sets):
                agreement += 1
            if any(winner in profile.approval_sets[voter] for voter in group):
                satisfaction += 1

        bound = (agreement * len(group)) // _worker_n
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


def find_pjr_violations_multiprocessing(
    instance: list[ApprovalProfile],
    decisions: list[Any],
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    """Same check as verify_pjr.find_pjr_violations (see there for the
    PJR definition), but splits the voter-subset enumeration across a
    process pool instead of checking groups one at a time.

    max_workers caps how many worker processes are used (default: all
    available CPU cores, via os.cpu_count()).
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
    if not all_groups:
        return []

    workers = max_workers if max_workers is not None else (os.cpu_count() or 1)
    workers = max(1, min(workers, len(all_groups)))

    # A handful of chunks per worker, so the pool can load-balance across
    # workers that finish early, rather than each worker getting one
    # fixed, possibly-uneven-sized slice.
    chunk_size = max(1, len(all_groups) // (workers * 4))
    chunks = [all_groups[i : i + chunk_size] for i in range(0, len(all_groups), chunk_size)]

    violations: list[dict[str, Any]] = []
    with multiprocessing.Pool(
        processes=workers, initializer=_init_worker, initargs=(instance, decisions)
    ) as pool:
        for chunk_violations in track(
            pool.imap_unordered(_check_groups_chunk, chunks),
            total=len(chunks),
            description=f"Verifying PJR ({workers} workers)...",
        ):
            violations.extend(chunk_violations)

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


def _format_metadata(metadata: dict[str, Any] | None) -> str:
    """Format a run's metadata as a trailing string for log/console
    lines, e.g. " (n=5, m=10, T=5, approval_threshold=1.5)". Empty
    string if there is no metadata to show.
    """
    if not metadata:
        return ""
    return " (" + ", ".join(f"{key}={value}" for key, value in metadata.items()) + ")"


def verify_run(run_dir: Path, max_workers: int | None = None) -> dict[str, Any] | None:
    """Load a run's approval profile and decision sequence, check PJR for
    every voter subset (via a process pool), and save the violations to
    run_dir/violations.jsonl.

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
    violations = find_pjr_violations_multiprocessing(instance, decisions, max_workers)

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


def verify_experiment(experiment_dir: Path, max_workers: int | None = None) -> None:
    """Verify PJR for every run_* subdirectory of experiment_dir, using a
    process pool for each run's verification.

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
    worst_overall: tuple[str, int, dict[str, Any], dict[str, Any] | None] | None = None

    for run_dir in run_dirs:
        summary = verify_run(run_dir, max_workers)
        if summary is None:
            continue

        total_violations += summary["num_violations"]
        if summary["num_violations"] > 0:
            num_runs_violating += 1

        worst = summary["worst"]
        if worst is not None:
            gap = worst["bound"] - worst["satisfaction"]
            if worst_overall is None or gap > worst_overall[1]:
                worst_overall = (run_dir.name, gap, worst, summary["metadata"])

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify PJR for a saved experiment, using a multiprocessing pool."
    )
    parser.add_argument(
        "experiment_dir", type=Path, help="path to an experiment directory (contains run_* dirs)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="maximum number of worker processes (default: all available CPU cores)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    verify_experiment(args.experiment_dir, args.max_workers)
