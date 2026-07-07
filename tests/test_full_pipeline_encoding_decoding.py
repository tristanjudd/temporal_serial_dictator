from pathlib import Path

from src.encoding.decoding import load_decisions_json, load_profile_jsonl
from src.encoding.encoding import save_decisions_json, save_profile_jsonl
from src.synthetic_data_tools.instances import generate_instance
from src.voting_rules.serial_dictator import SerialDictator

from .test_encoding_decoding_roundtrip import assert_profiles_equal


def test_generate_run_encode_and_decode_round_trip(tmp_path: Path) -> None:
    n = 5
    m = 4
    T = 10

    instance = generate_instance(
        n=n,
        m=m,
        T=T,
        sigma=0.2,
        voter_point_mode="normal",
        cand_point_mode="normal",
        approval_threshold=1.5,
    )

    serial_dictator: SerialDictator[int, int] = SerialDictator(voters=list(range(n)))
    decisions = serial_dictator(instance)

    profile_path = tmp_path / "instance.jsonl"
    decisions_path = tmp_path / "decisions.json"
    save_profile_jsonl(instance, profile_path)
    save_decisions_json(decisions, decisions_path)

    decoded_instance = load_profile_jsonl(profile_path)
    decoded_decisions = load_decisions_json(decisions_path)

    assert_profiles_equal(instance, decoded_instance)
    assert decoded_decisions == decisions
