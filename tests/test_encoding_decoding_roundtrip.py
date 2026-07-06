from pathlib import Path

from src.encoding.decoding import load_profile_jsonl
from src.encoding.encoding import save_profile_jsonl
from src.synthetic_data.instances import generate_instance


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

    assert decoded is not None
    assert len(decoded) == len(instance)
    for original_round, decoded_round in zip(instance, decoded, strict=True):
        assert list(original_round.voters) == list(decoded_round.voters)
        assert list(original_round.cands) == list(decoded_round.cands)
        assert {
            voter: set(approved) for voter, approved in original_round.approval_sets.items()
        } == {voter: set(approved) for voter, approved in decoded_round.approval_sets.items()}
