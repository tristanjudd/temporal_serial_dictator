import json
from pathlib import Path

from src.synthetic_data_tools.run_synthetic_experiment import run_synthetic_experiment


def test_creates_expected_files_and_metadata(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "exp"
    run_dirs = run_synthetic_experiment(
        T=4, n=5, m=3, approval_threshold=2.0, num_experiments=3, experiment_dir=experiment_dir
    )

    assert run_dirs == [experiment_dir / f"run_{i}" for i in range(3)]
    for run_dir in run_dirs:
        assert run_dir is not None
        assert (run_dir / "approvals.jsonl").exists()
        assert (run_dir / "decisions.json").exists()
        metadata = json.loads((run_dir / "metadata.json").read_text())
        assert metadata["T"] == 4
        assert metadata["n"] == 5
        assert metadata["m"] == 3
        assert metadata["approval_threshold"] == 2.0
        assert isinstance(metadata["seed"], int)


def test_same_seed_reproduces_identical_output(tmp_path: Path) -> None:
    run_synthetic_experiment(
        T=5,
        n=6,
        m=4,
        approval_threshold=1.5,
        num_experiments=2,
        seed=123,
        experiment_dir=tmp_path / "a",
    )
    run_synthetic_experiment(
        T=5,
        n=6,
        m=4,
        approval_threshold=1.5,
        num_experiments=2,
        seed=123,
        experiment_dir=tmp_path / "b",
    )

    for run_name in ["run_0", "run_1"]:
        for filename in ["approvals.jsonl", "decisions.json", "metadata.json"]:
            a = (tmp_path / "a" / run_name / filename).read_bytes()
            b = (tmp_path / "b" / run_name / filename).read_bytes()
            assert a == b, f"{run_name}/{filename} differs between identical-seed runs"


def test_different_seed_gives_different_output(tmp_path: Path) -> None:
    run_synthetic_experiment(
        T=5,
        n=6,
        m=4,
        approval_threshold=1.5,
        num_experiments=1,
        seed=1,
        experiment_dir=tmp_path / "a",
    )
    run_synthetic_experiment(
        T=5,
        n=6,
        m=4,
        approval_threshold=1.5,
        num_experiments=1,
        seed=2,
        experiment_dir=tmp_path / "b",
    )

    approvals_a = (tmp_path / "a" / "run_0" / "approvals.jsonl").read_bytes()
    approvals_b = (tmp_path / "b" / "run_0" / "approvals.jsonl").read_bytes()
    assert approvals_a != approvals_b


def test_omitted_seed_is_still_recorded_and_differs_between_calls(tmp_path: Path) -> None:
    run_synthetic_experiment(
        T=3, n=4, m=3, approval_threshold=1.5, num_experiments=1, experiment_dir=tmp_path / "a"
    )
    run_synthetic_experiment(
        T=3, n=4, m=3, approval_threshold=1.5, num_experiments=1, experiment_dir=tmp_path / "b"
    )

    seed_a = json.loads((tmp_path / "a" / "run_0" / "metadata.json").read_text())["seed"]
    seed_b = json.loads((tmp_path / "b" / "run_0" / "metadata.json").read_text())["seed"]
    assert isinstance(seed_a, int)
    assert seed_a != seed_b


def test_run_offset_shifts_run_directory_names(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "exp"
    run_dirs = run_synthetic_experiment(
        T=3, n=4, m=3, num_experiments=2, experiment_dir=experiment_dir, run_offset=10
    )

    assert run_dirs == [experiment_dir / "run_10", experiment_dir / "run_11"]


def test_multiple_calls_can_share_one_experiment_dir(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "shared"
    first = run_synthetic_experiment(
        T=3, n=4, m=3, num_experiments=2, experiment_dir=experiment_dir, run_offset=0
    )
    second = run_synthetic_experiment(
        T=5, n=8, m=5, num_experiments=2, experiment_dir=experiment_dir, run_offset=2
    )

    all_run_dirs = sorted(p.name for p in experiment_dir.iterdir() if p.name.startswith("run_"))
    assert all_run_dirs == ["run_0", "run_1", "run_2", "run_3"]
    assert first == [experiment_dir / "run_0", experiment_dir / "run_1"]
    assert second == [experiment_dir / "run_2", experiment_dir / "run_3"]
