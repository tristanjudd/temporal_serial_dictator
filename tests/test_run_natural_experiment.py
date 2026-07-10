from pathlib import Path

from src.data_transformation.tsoi_to_json import tsoi_dir_to_json
from src.encoding.decoding import load_decisions_json, load_profile_jsonl
from src.real_data_tools import run_natural_experiment as run_natural_experiment_module
from src.real_data_tools.run_natural_experiment import load_jsonl_dataset, run_natural_experiment
from src.voting_rules.serial_dictator import SerialDictator

ROUND_1 = """3
101,Candidate A
102,Candidate B
103,Candidate C
2,2,2
alice:1,101[10],102[5]
bob:1,102[8],103[3]
"""

# bob does not participate in this round -- exercises the missing-voter backfill.
ROUND_2 = """3
101,Candidate A
102,Candidate B
104,Candidate D
1,1,1
alice:1,104[7],101[2]
"""


def _build_dummy_jsonl(tmp_path: Path, name: str = "dataset.jsonl") -> Path:
    tsoi_dir = tmp_path / "tsoi_src"
    tsoi_dir.mkdir(parents=True, exist_ok=True)
    (tsoi_dir / "round_1.tsoi").write_text(ROUND_1)
    (tsoi_dir / "round_2.tsoi").write_text(ROUND_2)

    jsonl_path = tmp_path / name
    tsoi_dir_to_json(tsoi_dir, jsonl_path)
    return jsonl_path


def test_load_jsonl_dataset_backfills_missing_voter_with_all_candidates(tmp_path: Path) -> None:
    jsonl_path = _build_dummy_jsonl(tmp_path)

    loaded = load_jsonl_dataset(jsonl_path)
    assert loaded is not None
    metadata, instance = loaded

    assert metadata["T"] == 2
    assert len(instance) == 2
    round_1, round_2 = instance

    assert list(round_1.voters) == ["alice", "bob"]
    assert list(round_2.voters) == ["alice", "bob"]

    # candidate IDs are the real IDs from the file, not remapped indices.
    assert round_1.cands == [101, 102, 103]
    assert round_2.cands == [101, 102, 104]

    assert round_1.approval_sets["alice"] == [101, 102]
    assert round_1.approval_sets["bob"] == [102, 103]

    # alice actually voted in round 2; bob didn't, so bob should be
    # backfilled as indifferent, i.e. approving all of round 2's candidates.
    assert round_2.approval_sets["alice"] == [104, 101]
    assert round_2.approval_sets["bob"] == round_2.cands


def test_serial_dictator_runs_on_loaded_natural_data(tmp_path: Path) -> None:
    jsonl_path = _build_dummy_jsonl(tmp_path)
    loaded = load_jsonl_dataset(jsonl_path)
    assert loaded is not None
    _, instance = loaded

    serial_dictator: SerialDictator[str, int] = SerialDictator(voters=["alice", "bob"])
    decisions = serial_dictator(instance)

    assert len(decisions) == 2
    for round_profile, winner in zip(instance, decisions, strict=True):
        assert winner in round_profile.cands


def test_run_natural_experiment_end_to_end(tmp_path: Path, monkeypatch) -> None:
    jsonl_path = _build_dummy_jsonl(tmp_path, name="dataset")
    experiments_dir = tmp_path / "experiments"
    monkeypatch.setattr(run_natural_experiment_module, "EXPERIMENTS_DIR", experiments_dir)

    run_dirs = run_natural_experiment(jsonl_path)

    assert run_dirs == [experiments_dir / "dataset" / "run_0"]
    run_dir = run_dirs[0]
    assert run_dir is not None

    instance = load_profile_jsonl(run_dir / "approvals.jsonl")
    decisions = load_decisions_json(run_dir / "decisions.json")

    assert instance is not None
    assert decisions is not None
    assert len(instance) == 2
    assert len(decisions) == 2
    assert list(instance[0].voters) == ["alice", "bob"]

    # running it again on the same dataset should not overwrite run_0.
    second_run_dirs = run_natural_experiment(jsonl_path)
    assert second_run_dirs == [experiments_dir / "dataset" / "run_1"]


# 5 voters across 2 rounds (with distinct candidate ids per round), used to
# exercise downsizing across multiple runs.
ROUND_1_FIVE_VOTERS = """2
101,Candidate A
102,Candidate B
5,5,5
v1:1,101[10]
v2:1,101[9]
v3:1,102[8]
v4:1,102[7]
v5:1,101[6]
"""

ROUND_2_FIVE_VOTERS = """2
201,Candidate X
202,Candidate Y
5,5,5
v1:1,201[5]
v2:1,202[4]
v3:1,201[3]
v4:1,202[2]
v5:1,201[1]
"""


def test_run_natural_experiment_downsize_preserves_master_order(
    tmp_path: Path, monkeypatch
) -> None:
    tsoi_dir = tmp_path / "tsoi_src"
    tsoi_dir.mkdir(parents=True, exist_ok=True)
    (tsoi_dir / "round_1.tsoi").write_text(ROUND_1_FIVE_VOTERS)
    (tsoi_dir / "round_2.tsoi").write_text(ROUND_2_FIVE_VOTERS)
    jsonl_path = tmp_path / "dataset"
    tsoi_dir_to_json(tsoi_dir, jsonl_path)

    experiments_dir = tmp_path / "experiments"
    monkeypatch.setattr(run_natural_experiment_module, "EXPERIMENTS_DIR", experiments_dir)

    master_order = ["v1", "v2", "v3", "v4", "v5"]
    run_dirs = run_natural_experiment(jsonl_path, downsize=True, sample_size=2, num_experiments=5)

    assert len(run_dirs) == 5
    for run_dir in run_dirs:
        assert run_dir is not None
        instance = load_profile_jsonl(run_dir / "approvals.jsonl")
        assert instance is not None

        sampled_voters = list(instance[0].voters)
        assert len(sampled_voters) == 2
        # whichever two voters were sampled, they appear in the same
        # relative order as in the full (master) dataset.
        expected_order = [voter for voter in master_order if voter in sampled_voters]
        assert sampled_voters == expected_order
