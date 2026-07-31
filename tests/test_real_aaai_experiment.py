import json
from pathlib import Path

from src.data_transformation.tsoi_to_json import tsoi_dir_to_json
from src.real_data_tools import real_aaai_experiment as real_aaai_experiment_module
from src.real_data_tools.real_aaai_experiment import run_real_aaai_experiment

# 5 voters, 2 rounds -- enough to exercise both the downsize (SAMPLE_SIZE < 5)
# and full-voter-set (SAMPLE_SIZE >= 5) paths by varying SAMPLE_SIZE alone.
ROUND_1 = """2
101,Candidate A
102,Candidate B
5,5,5
v1:1,101[10]
v2:1,101[9]
v3:1,102[8]
v4:1,102[7]
v5:1,101[6]
"""

ROUND_2 = """2
201,Candidate X
202,Candidate Y
5,5,5
v1:1,201[5]
v2:1,202[4]
v3:1,201[3]
v4:1,202[2]
v5:1,201[1]
"""


def _build_dataset(tmp_path: Path, name: str) -> Path:
    tsoi_dir = tmp_path / f"{name}_src"
    tsoi_dir.mkdir(parents=True, exist_ok=True)
    (tsoi_dir / "round_1.tsoi").write_text(ROUND_1)
    (tsoi_dir / "round_2.tsoi").write_text(ROUND_2)

    datasets_dir = tmp_path / "json_datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = datasets_dir / name
    tsoi_dir_to_json(tsoi_dir, jsonl_path)
    return jsonl_path


def test_downsizes_when_sample_size_below_voter_count(tmp_path: Path, monkeypatch) -> None:
    datasets_dir = tmp_path / "json_datasets"
    _build_dataset(tmp_path, "mydataset")
    experiments_dir = tmp_path / "experiments"

    monkeypatch.setattr(real_aaai_experiment_module, "DATASETS_DIR", datasets_dir)
    monkeypatch.setattr(real_aaai_experiment_module, "EXPERIMENTS_DIR", experiments_dir)
    monkeypatch.setattr(real_aaai_experiment_module, "SAMPLE_SIZE", 2)

    run_real_aaai_experiment(num_experiments=3)

    dataset_dir = experiments_dir / "mydataset"
    run_dirs = sorted(p.name for p in dataset_dir.iterdir() if p.name.startswith("run_"))
    assert run_dirs == ["run_0", "run_1", "run_2"]

    for run_name in run_dirs:
        metadata = json.loads((dataset_dir / run_name / "metadata.json").read_text())
        assert metadata["downsized"] is True
        assert metadata["n"] == 2
        assert metadata["original_n"] == 5


def test_uses_full_voter_set_when_sample_size_at_or_above_voter_count(
    tmp_path: Path, monkeypatch
) -> None:
    datasets_dir = tmp_path / "json_datasets"
    _build_dataset(tmp_path, "mydataset")
    experiments_dir = tmp_path / "experiments"

    monkeypatch.setattr(real_aaai_experiment_module, "DATASETS_DIR", datasets_dir)
    monkeypatch.setattr(real_aaai_experiment_module, "EXPERIMENTS_DIR", experiments_dir)
    monkeypatch.setattr(real_aaai_experiment_module, "SAMPLE_SIZE", 10)

    # num_experiments should be ignored when there's nothing to downsize:
    # exactly one run on the full voter set regardless.
    run_real_aaai_experiment(num_experiments=5)

    dataset_dir = experiments_dir / "mydataset"
    run_dirs = sorted(p.name for p in dataset_dir.iterdir() if p.name.startswith("run_"))
    assert run_dirs == ["run_0"]

    metadata = json.loads((dataset_dir / "run_0" / "metadata.json").read_text())
    assert metadata["downsized"] is False
    assert metadata["n"] == 5
    assert metadata["original_n"] == 5


def test_manifest_and_summary_reflect_all_runs(tmp_path: Path, monkeypatch) -> None:
    datasets_dir = tmp_path / "json_datasets"
    _build_dataset(tmp_path, "mydataset")
    experiments_dir = tmp_path / "experiments"

    monkeypatch.setattr(real_aaai_experiment_module, "DATASETS_DIR", datasets_dir)
    monkeypatch.setattr(real_aaai_experiment_module, "EXPERIMENTS_DIR", experiments_dir)
    monkeypatch.setattr(real_aaai_experiment_module, "SAMPLE_SIZE", 2)

    run_real_aaai_experiment(num_experiments=2)

    dataset_dir = experiments_dir / "mydataset"
    manifest_lines = (dataset_dir / "experiment_manifest.jsonl").read_text().splitlines()
    assert len(manifest_lines) == 2

    summary_text = (dataset_dir / "experiment_summary.log").read_text()
    assert "Total runs: 2" in summary_text
    assert "Downsized runs: 2/2" in summary_text


def test_one_bad_dataset_does_not_abort_the_others(tmp_path: Path, monkeypatch) -> None:
    datasets_dir = tmp_path / "json_datasets"
    _build_dataset(tmp_path, "good_dataset")
    experiments_dir = tmp_path / "experiments"

    monkeypatch.setattr(real_aaai_experiment_module, "DATASETS_DIR", datasets_dir)
    monkeypatch.setattr(real_aaai_experiment_module, "EXPERIMENTS_DIR", experiments_dir)
    monkeypatch.setattr(real_aaai_experiment_module, "SAMPLE_SIZE", 2)

    # a second "dataset" file that isn't valid jsonl at all -- load_jsonl_dataset
    # will fail to parse it and return None, which _run_dataset already
    # handles gracefully; this exercises that _run_dataset failing outright
    # (raising) for one dataset still lets others proceed.
    original_run_dataset = real_aaai_experiment_module._run_dataset

    def flaky_run_dataset(dataset_path: Path, num_experiments: int) -> None:
        if dataset_path.name == "bad_dataset":
            raise RuntimeError("simulated failure")
        original_run_dataset(dataset_path, num_experiments)

    (datasets_dir / "bad_dataset").write_text("not valid jsonl\n")
    monkeypatch.setattr(real_aaai_experiment_module, "_run_dataset", flaky_run_dataset)

    run_real_aaai_experiment(num_experiments=1)

    assert (experiments_dir / "good_dataset" / "run_0").exists()
    assert not (experiments_dir / "bad_dataset").exists()
