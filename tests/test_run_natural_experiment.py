from pathlib import Path

from src.encoding.decoding import load_decisions_json, load_profile_jsonl
from src.real_data_tools import run_natural_experiment as run_natural_experiment_module
from src.real_data_tools.run_natural_experiment import run_natural_experiment
from src.real_data_tools.tsoi_parser import parse_tsoi_directory
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


def _write_dummy_dataset(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "round_1.tsoi").write_text(ROUND_1)
    (directory / "round_2.tsoi").write_text(ROUND_2)
    return directory


def test_parse_tsoi_directory_backfills_missing_voter_with_all_candidates(tmp_path: Path) -> None:
    data_dir = _write_dummy_dataset(tmp_path)

    instance = parse_tsoi_directory(data_dir)

    assert len(instance) == 2
    round_1, round_2 = instance

    assert list(round_1.voters) == ["alice", "bob"]
    assert list(round_2.voters) == ["alice", "bob"]

    assert round_1.approval_sets["alice"] == [0, 1]
    assert round_1.approval_sets["bob"] == [1, 2]

    # alice actually voted in round 2; bob didn't, so bob should be
    # backfilled as indifferent, i.e. approving all of round 2's candidates.
    assert round_2.approval_sets["alice"] == [2, 0]
    assert round_2.approval_sets["bob"] == round_2.cands


def test_serial_dictator_runs_on_parsed_natural_data(tmp_path: Path) -> None:
    data_dir = _write_dummy_dataset(tmp_path)
    instance = parse_tsoi_directory(data_dir)

    serial_dictator: SerialDictator[str, int] = SerialDictator(voters=["alice", "bob"])
    decisions = serial_dictator(instance)

    assert len(decisions) == 2
    for round_profile, winner in zip(instance, decisions, strict=True):
        assert winner in round_profile.cands


def test_run_natural_experiment_end_to_end(tmp_path: Path, monkeypatch) -> None:
    data_dir = _write_dummy_dataset(tmp_path / "dataset")
    experiments_dir = tmp_path / "experiments"
    monkeypatch.setattr(run_natural_experiment_module, "EXPERIMENTS_DIR", experiments_dir)

    run_dir = run_natural_experiment(data_dir)

    assert run_dir is not None
    assert run_dir == experiments_dir / "dataset" / "run_0"

    instance = load_profile_jsonl(run_dir / "approvals.jsonl")
    decisions = load_decisions_json(run_dir / "decisions.json")

    assert instance is not None
    assert decisions is not None
    assert len(instance) == 2
    assert len(decisions) == 2
    assert list(instance[0].voters) == ["alice", "bob"]

    # running it again on the same dataset should not overwrite run_0.
    second_run_dir = run_natural_experiment(data_dir)
    assert second_run_dir == experiments_dir / "dataset" / "run_1"
