"""Collate PJR violations across every run of a verified experiment.

Reads each run_*/violations.jsonl (as produced by verify_pjr.py /
multiprocessing_verify_pjr.py), annotates every violation with the run it
came from and that run's configuration (metadata.json, if present), and
writes them all into a single all_violations.jsonl in the experiment
directory -- so violations across many runs and configurations can be
analyzed together (e.g. loaded into pandas) instead of picked apart
run-by-run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


def _load_metadata(run_dir: Path) -> dict[str, Any]:
    """Load run_dir/metadata.json if it exists, else {}.

    A parse failure is reported as a warning and treated the same as a
    missing file -- the run's violations are still collated, just
    without configuration info attached.
    """
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        metadata: dict[str, Any] = json.loads(metadata_path.read_text())
        return metadata
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: could not read metadata from '{metadata_path}': {e}", file=sys.stderr)
        return {}


def collate_violations(experiment_dir: Path) -> Path | None:
    """Collate every run_*/violations.jsonl under experiment_dir into a
    single experiment_dir/all_violations.jsonl.

    Each collated entry is a violation dict extended with "run" (the run
    directory name) and that run's metadata.json fields, if any. Runs
    without a violations.jsonl (i.e. not yet verified) are skipped.

    Returns the path to the collated file, or None if experiment_dir is
    invalid or the file could not be written.
    """
    if not experiment_dir.is_dir():
        print(f"Error: '{experiment_dir}' is not a directory.", file=sys.stderr)
        return None

    run_dirs = sorted(
        (p for p in experiment_dir.iterdir() if p.is_dir() and p.name.startswith("run_")),
        key=lambda p: int(p.name.removeprefix("run_")),
    )
    if not run_dirs:
        print(f"Error: no run_* subdirectories found in '{experiment_dir}'.", file=sys.stderr)
        return None

    collated: list[dict[str, Any]] = []
    num_runs_seen = 0
    for run_dir in run_dirs:
        violations_path = run_dir / "violations.jsonl"
        if not violations_path.exists():
            continue
        num_runs_seen += 1

        metadata = _load_metadata(run_dir)
        try:
            with violations_path.open() as f:
                for raw_line in f:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    violation = json.loads(raw_line)
                    collated.append({"run": run_dir.name, **metadata, **violation})
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: could not read '{violations_path}': {e}", file=sys.stderr)

    output_path = experiment_dir / "all_violations.jsonl"
    try:
        with output_path.open("w") as f:
            for entry in collated:
                f.write(json.dumps(entry) + "\n")
    except OSError as e:
        print(f"Error writing collated violations to '{output_path}': {e}", file=sys.stderr)
        return None

    console.print(
        f"[bold]Collated {len(collated)} violation(s) from {num_runs_seen}/{len(run_dirs)} "
        f"verified run(s) into[/bold] {output_path}"
    )
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collate all runs' PJR violations into a single JSONL file."
    )
    parser.add_argument(
        "experiment_dir", type=Path, help="path to an experiment directory (contains run_* dirs)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    collate_violations(args.experiment_dir)
