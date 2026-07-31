import json
from pathlib import Path

from src.verification.collate_violations import collate_violations


def _write_violations(run_dir: Path, violations: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "violations.jsonl").open("w") as f:
        for v in violations:
            f.write(json.dumps(v) + "\n")


def _read_entries(output_path: Path | None) -> list[dict]:
    assert output_path is not None
    return [json.loads(line) for line in output_path.read_text().splitlines()]


def test_collates_violations_with_metadata(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "exp"
    run0 = experiment_dir / "run_0"
    _write_violations(run0, [{"voters": [0, 1], "agreement": 2, "bound": 1, "satisfaction": 0}])
    (run0 / "metadata.json").write_text(json.dumps({"n": 5, "T": 10}))

    output_path = collate_violations(experiment_dir)

    assert output_path == experiment_dir / "all_violations.jsonl"
    entries = _read_entries(output_path)
    assert entries == [
        {
            "run": "run_0",
            "n": 5,
            "T": 10,
            "voters": [0, 1],
            "agreement": 2,
            "bound": 1,
            "satisfaction": 0,
        }
    ]


def test_run_without_metadata_still_collates_violations(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "exp"
    run0 = experiment_dir / "run_0"
    _write_violations(run0, [{"voters": [0], "agreement": 1, "bound": 1, "satisfaction": 0}])
    # deliberately no metadata.json

    output_path = collate_violations(experiment_dir)

    entries = _read_entries(output_path)
    assert entries == [
        {"run": "run_0", "voters": [0], "agreement": 1, "bound": 1, "satisfaction": 0}
    ]


def test_run_without_violations_file_is_skipped_not_counted(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "exp"
    (experiment_dir / "run_0").mkdir(parents=True)  # no violations.jsonl -- not yet verified
    run1 = experiment_dir / "run_1"
    _write_violations(run1, [{"voters": [0], "agreement": 1, "bound": 1, "satisfaction": 0}])

    output_path = collate_violations(experiment_dir)

    entries = _read_entries(output_path)
    assert len(entries) == 1
    assert entries[0]["run"] == "run_1"


def test_satisfied_run_with_empty_violations_file_contributes_nothing(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "exp"
    _write_violations(experiment_dir / "run_0", [])  # PJR satisfied: empty violations.jsonl

    output_path = collate_violations(experiment_dir)

    assert output_path is not None
    assert output_path.read_text() == ""


def test_nonexistent_experiment_dir_returns_none(tmp_path: Path) -> None:
    assert collate_violations(tmp_path / "does-not-exist") is None


def test_experiment_dir_with_no_run_subdirs_returns_none(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "exp"
    experiment_dir.mkdir()
    assert collate_violations(experiment_dir) is None


def test_multiple_runs_collate_in_run_order(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "exp"
    for i in range(3):
        run_dir = experiment_dir / f"run_{i}"
        _write_violations(run_dir, [{"voters": [i], "agreement": 1, "bound": 1, "satisfaction": 0}])
        (run_dir / "metadata.json").write_text(json.dumps({"n": i}))

    output_path = collate_violations(experiment_dir)

    entries = _read_entries(output_path)
    assert [e["run"] for e in entries] == ["run_0", "run_1", "run_2"]
