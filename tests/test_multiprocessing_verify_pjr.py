from src.synthetic_data_tools.profiles import ApprovalProfile
from src.verification.multiprocessing_verify_pjr import find_pjr_violations_multiprocessing
from src.verification.verify_pjr import find_pjr_violations


def _as_set(violations):
    return {(tuple(v["voters"]), v["agreement"], v["bound"], v["satisfaction"]) for v in violations}


def test_satisfies_pjr_when_everyone_agrees_and_wins():
    instance = [
        ApprovalProfile(voters=[0, 1, 2], cands=[0, 1], approval_sets={0: [0], 1: [0], 2: [0]}),
        ApprovalProfile(voters=[0, 1, 2], cands=[0, 1], approval_sets={0: [1], 1: [1], 2: [1]}),
    ]
    decisions = [0, 1]

    assert find_pjr_violations_multiprocessing(instance, decisions, max_workers=2) == []


def test_satisfies_pjr_with_default_max_workers():
    # exercises the max_workers=None path (falls back to os.cpu_count()).
    voters = [0, 1, 2, 3]
    cands = [0, 1, 2, 3]
    approval_sets = {v: [v] for v in voters}
    instance = [ApprovalProfile(voters, cands, approval_sets) for _ in range(4)]
    decisions = [0, 1, 2, 3]

    assert find_pjr_violations_multiprocessing(instance, decisions) == []


def test_matches_sequential_implementation_on_violating_instances():
    cases = [
        (
            [ApprovalProfile([0, 1, 2], [0, 1], {0: [1], 1: [1], 2: [0]}) for _ in range(4)],
            [0, 0, 0, 0],
        ),
        (
            [
                ApprovalProfile([0, 1, 2, 3], [0, 1, 2, 3], {0: [3], 1: [3], 2: [3], 3: [0]})
                for _ in range(8)
            ],
            [0] * 8,
        ),
    ]
    for instance, decisions in cases:
        sequential = _as_set(find_pjr_violations(instance, decisions))
        parallel = _as_set(find_pjr_violations_multiprocessing(instance, decisions, max_workers=2))

        assert parallel == sequential
        assert len(parallel) > 0  # sanity: these cases really do have violations
