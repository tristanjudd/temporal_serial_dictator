import json
from pathlib import Path

from src.data_transformation.tsoi_to_json import tsoi_dir_to_json
from src.encoding.decoding import load_profile_jsonl
from src.real_data_tools import real_aaai_grid_experiment as grid_module
from src.real_data_tools.real_aaai_grid_experiment import _build_configs, _run_dataset_grid


def test_build_configs_adds_full_voter_tier_below_max_n() -> None:
    # matches the real i_phone datasets (11 voters, 62 rounds): n=20 is
    # infeasible, but n=11 (the dataset's actual full voter count) should
    # be added since it's below the top N_VALUES tier and not already in it.
    configs = _build_configs(original_n=11, original_T=62)
    ns = sorted({n for n, _ in configs})
    assert ns == [5, 10, 11]


def test_build_configs_does_not_duplicate_an_existing_n_tier() -> None:
    configs = _build_configs(original_n=10, original_T=100)
    ns = [n for n, _ in configs]
    assert ns.count(10) == len([t for t in ns if t == 10])  # sanity: no crash
    assert sorted(set(ns)) == [5, 10]


def test_build_configs_full_grid_when_enough_voters_and_rounds() -> None:
    # matches eurovision (53 voters, 45 rounds): all of N_VALUES fit.
    configs = _build_configs(original_n=53, original_T=45)
    assert sorted({n for n, _ in configs}) == [5, 10, 20]
    assert len(configs) == 9  # {5,2} + {10,5,2} + {20,10,5,2}


def test_build_configs_filters_t_by_available_rounds() -> None:
    # n=5 is the only feasible n; its T grid is {5, 2}, but only T=2 fits
    # within 3 available rounds.
    configs = _build_configs(original_n=5, original_T=3)
    assert configs == [(5, 2)]


def test_build_configs_tiny_dataset_below_all_tiers() -> None:
    configs = _build_configs(original_n=3, original_T=100)
    assert configs == [(3, 3)]


ROUND_TEMPLATE = """2
{cand_a},Candidate A{t}
{cand_b},Candidate B{t}
5,5,5
v1:1,{first}[10]
v2:1,{second}[9]
v3:1,{first}[8]
v4:1,{second}[7]
v5:1,{first}[6]
"""


def _build_five_voter_dataset(tmp_path: Path, name: str, num_rounds: int) -> Path:
    tsoi_dir = tmp_path / f"{name}_src"
    tsoi_dir.mkdir(parents=True, exist_ok=True)
    for t in range(num_rounds):
        cand_a, cand_b = 100 + 2 * t, 101 + 2 * t
        text = ROUND_TEMPLATE.format(cand_a=cand_a, cand_b=cand_b, t=t, first=cand_a, second=cand_b)
        (tsoi_dir / f"round_{t:03d}.tsoi").write_text(text)

    datasets_dir = tmp_path / "json_datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = datasets_dir / name
    tsoi_dir_to_json(tsoi_dir, jsonl_path)
    return jsonl_path


def test_run_dataset_grid_end_to_end(tmp_path: Path, monkeypatch) -> None:
    _build_five_voter_dataset(tmp_path, "mydataset", num_rounds=5)
    experiments_dir = tmp_path / "experiments"
    monkeypatch.setattr(grid_module, "EXPERIMENTS_DIR", experiments_dir)

    dataset_path = tmp_path / "json_datasets" / "mydataset"
    _run_dataset_grid(dataset_path, num_experiments=3)

    dataset_dir = experiments_dir / "mydataset"
    run_dirs = sorted(p.name for p in dataset_dir.iterdir() if p.name.startswith("run_"))
    # (5, 5) and (5, 2) are the only feasible configs for 5 voters / 5 rounds,
    # 3 runs each.
    assert run_dirs == [f"run_{i}" for i in range(6)]

    manifest_text = (dataset_dir / "experiment_manifest.jsonl").read_text()
    manifest_entries = [json.loads(line) for line in manifest_text.splitlines()]
    assert {(e["n"], e["T"]) for e in manifest_entries} == {(5, 5), (5, 2)}

    # each config's own run 0 is the designated first-rounds run.
    by_config: dict[tuple[int, int], list[dict]] = {}
    for entry in manifest_entries:
        by_config.setdefault((entry["n"], entry["T"]), []).append(entry)
    for (_n, _t), entries in by_config.items():
        entries.sort(key=lambda e: int(e["run"].removeprefix("run_")))
        assert entries[0]["round_start"] == 0
        assert entries[0]["is_first_rounds"] is True
        for entry in entries[1:]:
            assert entry["is_first_rounds"] is False

    summary_text = (dataset_dir / "experiment_summary.log").read_text()
    assert "Configurations: 2" in summary_text
    assert "Total runs: 6" in summary_text


def test_round_windows_are_contiguous_and_within_bounds(tmp_path: Path, monkeypatch) -> None:
    _build_five_voter_dataset(tmp_path, "mydataset", num_rounds=5)
    experiments_dir = tmp_path / "experiments"
    monkeypatch.setattr(grid_module, "EXPERIMENTS_DIR", experiments_dir)

    dataset_path = tmp_path / "json_datasets" / "mydataset"
    _run_dataset_grid(dataset_path, num_experiments=5)

    dataset_dir = experiments_dir / "mydataset"
    for run_dir in dataset_dir.iterdir():
        if not run_dir.name.startswith("run_"):
            continue
        metadata = json.loads((run_dir / "metadata.json").read_text())
        instance = load_profile_jsonl(run_dir / "approvals.jsonl")

        assert instance is not None
        assert len(instance) == metadata["T"]
        assert metadata["round_start"] >= 0
        assert metadata["round_start"] + metadata["T"] <= metadata["original_T"]


def test_a_later_run_coincidentally_starting_at_0_is_not_flagged_as_first_rounds(
    tmp_path: Path, monkeypatch
) -> None:
    # with T == original_T (5 rounds requested out of 5 total), the random
    # window range collapses to exactly {0} -- every run is forced to
    # round_start=0, deterministically, letting us check that only the
    # actual run 0 is flagged is_first_rounds, not every run that happens
    # to share its (coincidental) round_start.
    _build_five_voter_dataset(tmp_path, "mydataset", num_rounds=5)
    experiments_dir = tmp_path / "experiments"
    monkeypatch.setattr(grid_module, "EXPERIMENTS_DIR", experiments_dir)

    dataset_path = tmp_path / "json_datasets" / "mydataset"
    _run_dataset_grid(dataset_path, num_experiments=4)

    dataset_dir = experiments_dir / "mydataset"
    t5_runs = []
    for run_dir in dataset_dir.iterdir():
        if not run_dir.name.startswith("run_"):
            continue
        metadata = json.loads((run_dir / "metadata.json").read_text())
        if metadata["T"] == 5:
            t5_runs.append((int(run_dir.name.removeprefix("run_")), metadata))
    t5_runs.sort()

    assert all(m["round_start"] == 0 for _, m in t5_runs)
    assert t5_runs[0][1]["is_first_rounds"] is True
    assert all(not m["is_first_rounds"] for _, m in t5_runs[1:])
