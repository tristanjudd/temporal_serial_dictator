from src.synthetic_data.instances import generate_instance
from src.voting_rules.serial_dictator import SerialDictator


def test_serial_dictator_runs_on_generated_instance():
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

    serial_dictator = SerialDictator(voters=list(range(n)))
    winners = serial_dictator(instance)

    assert len(winners) == T
    for round_profile, winner in zip(instance, winners, strict=True):
        assert winner in round_profile.cands
