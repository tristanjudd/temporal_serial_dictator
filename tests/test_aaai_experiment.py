import json
from pathlib import Path

from src.synthetic_data_tools import aaai_experiment as aaai_experiment_module
from src.synthetic_data_tools.aaai_experiment import build_grid, run_aaai_experiment


def test_build_grid_excludes_t_equal_1_and_dedupes_per_n() -> None:
    configs = build_grid()

    assert all(config.T > 1 for config in configs)

    t_values_by_n: dict[int, set[int]] = {}
    for config in configs:
        t_values_by_n.setdefault(config.n, set()).add(config.T)

    # n=5: divisors [1,2,4,8] -> 5,2,1,0 -> floor+min1 -> {5,2,1}, T=1 excluded -> {5,2}
    assert t_values_by_n[5] == {5, 2}
    # n=10: -> {10,5,2,1}, T=1 excluded -> {10,5,2}
    assert t_values_by_n[10] == {10, 5, 2}
    # n=20: -> {20,10,5,2}, T=1 excluded -> {20,10,5,2}
    assert t_values_by_n[20] == {20, 10, 5, 2}


def test_build_grid_covers_full_n_m_threshold_cross_product() -> None:
    configs = build_grid()
    # 2 T-values for n=5 ({5,2}), 3 for n=10 ({10,5,2}), 4 for n=20
    # ({20,10,5,2}) = 9 (n, T) pairs, times 3 m-values times 4 thresholds.
    assert len(configs) == 9 * 3 * 4


def test_run_aaai_experiment_end_to_end_with_small_grid(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(aaai_experiment_module, "EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(aaai_experiment_module, "N_VALUES", [5])
    monkeypatch.setattr(aaai_experiment_module, "T_DIVISORS", [1, 2])
    monkeypatch.setattr(aaai_experiment_module, "M_VALUES", [3])
    monkeypatch.setattr(aaai_experiment_module, "THRESHOLDS", [1.5])

    parent_dir = run_aaai_experiment(num_experiments=2, seed=42)

    assert parent_dir is not None
    assert parent_dir.parent == tmp_path
    # n=5 with divisors [1,2] -> T values {5, 2}, 2 configs x 2 runs = 4 runs
    run_dirs = sorted(p.name for p in parent_dir.iterdir() if p.name.startswith("run_"))
    assert run_dirs == ["run_0", "run_1", "run_2", "run_3"]

    manifest_text = (parent_dir / "experiment_manifest.jsonl").read_text()
    manifest_entries = [json.loads(line) for line in manifest_text.splitlines()]
    assert len(manifest_entries) == 4
    assert {entry["T"] for entry in manifest_entries} == {5, 2}
    assert all(entry["n"] == 5 and entry["m"] == 3 for entry in manifest_entries)

    assert (parent_dir / "experiment_summary.log").exists()
    summary_text = (parent_dir / "experiment_summary.log").read_text()
    assert "Seed: 42" in summary_text


def test_run_aaai_experiment_same_seed_is_reproducible(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(aaai_experiment_module, "EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(aaai_experiment_module, "N_VALUES", [5])
    monkeypatch.setattr(aaai_experiment_module, "T_DIVISORS", [1])
    monkeypatch.setattr(aaai_experiment_module, "M_VALUES", [3])
    monkeypatch.setattr(aaai_experiment_module, "THRESHOLDS", [1.5])

    dir_a = run_aaai_experiment(num_experiments=1, seed=7)
    dir_b = run_aaai_experiment(num_experiments=1, seed=7)
    assert dir_a is not None and dir_b is not None

    for filename in ["approvals.jsonl", "decisions.json"]:
        a = (dir_a / "run_0" / filename).read_bytes()
        b = (dir_b / "run_0" / filename).read_bytes()
        assert a == b
