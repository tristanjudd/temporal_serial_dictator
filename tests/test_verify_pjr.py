from src.synthetic_data.profiles import ApprovalProfile
from src.verification.verify_pjr import find_pjr_violations


def test_satisfies_pjr_when_everyone_agrees_and_wins():
    instance = [
        ApprovalProfile(voters=[0, 1, 2], cands=[0, 1], approval_sets={0: [0], 1: [0], 2: [0]}),
        ApprovalProfile(voters=[0, 1, 2], cands=[0, 1], approval_sets={0: [1], 1: [1], 2: [1]}),
    ]
    decisions = [0, 1]

    assert find_pjr_violations(instance, decisions) == []


def test_satisfies_pjr_with_disjoint_preferences_and_round_robin_winners():
    voters = [0, 1, 2, 3]
    cands = [0, 1, 2, 3]
    approval_sets = {v: [v] for v in voters}
    instance = [ApprovalProfile(voters, cands, approval_sets) for _ in range(4)]
    decisions = [0, 1, 2, 3]

    assert find_pjr_violations(instance, decisions) == []


def test_violates_pjr_when_agreeing_minority_is_ignored():
    voters = [0, 1, 2]
    cands = [0, 1]
    approval_sets = {0: [1], 1: [1], 2: [0]}
    instance = [ApprovalProfile(voters, cands, approval_sets) for _ in range(4)]
    decisions = [0, 0, 0, 0]

    violations = find_pjr_violations(instance, decisions)
    violating_groups = {tuple(v["voters"]) for v in violations}

    assert (0,) in violating_groups
    assert (1,) in violating_groups
    assert (0, 1) in violating_groups
    assert (2,) not in violating_groups


def test_violates_pjr_with_escalating_group_sizes():
    voters = [0, 1, 2, 3]
    cands = [0, 1, 2, 3]
    approval_sets = {0: [3], 1: [3], 2: [3], 3: [0]}
    instance = [ApprovalProfile(voters, cands, approval_sets) for _ in range(8)]
    decisions = [0] * 8

    violations = find_pjr_violations(instance, decisions)
    violating = {tuple(v["voters"]): v for v in violations}

    assert violating[(0,)]["bound"] - violating[(0,)]["satisfaction"] == 2
    assert violating[(0, 1)]["bound"] - violating[(0, 1)]["satisfaction"] == 4
    assert violating[(0, 1, 2)]["bound"] - violating[(0, 1, 2)]["satisfaction"] == 6
    assert (3,) not in violating
