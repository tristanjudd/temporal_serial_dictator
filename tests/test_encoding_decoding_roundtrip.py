from pathlib import Path

from src.encoding.decoding import load_profile_jsonl
from src.encoding.encoding import save_profile_jsonl
from src.synthetic_data.instances import generate_instance
from src.synthetic_data.profiles import ApprovalProfile


def assert_profiles_equal(
    expected: list[ApprovalProfile], actual: list[ApprovalProfile] | None
) -> None:
    assert actual is not None
    assert len(actual) == len(expected)
    for expected_round, actual_round in zip(expected, actual, strict=True):
        assert list(expected_round.voters) == list(actual_round.voters)
        assert list(expected_round.cands) == list(actual_round.cands)
        assert {
            voter: set(approved) for voter, approved in expected_round.approval_sets.items()
        } == {voter: set(approved) for voter, approved in actual_round.approval_sets.items()}


def test_encode_then_decode_reproduces_the_instance(tmp_path: Path) -> None:
    instance = generate_instance(
        n=4,
        m=3,
        T=5,
        sigma=0.2,
        voter_point_mode="normal",
        cand_point_mode="normal",
        approval_threshold=1.5,
    )

    path = tmp_path / "instance.jsonl"
    save_profile_jsonl(instance, path)
    decoded = load_profile_jsonl(path)

    assert_profiles_equal(instance, decoded)
