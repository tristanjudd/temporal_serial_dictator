import random

import pytest

from src.synthetic_data_tools.points import generate_2d_points

ALL_MODES = [
    "eucl1",
    "eucl2",
    "eucl3",
    "eucl4",
    "eucl5",
    "eucl6",
    "eucl2plus",
    "normal",
    "uniform_square",
]


@pytest.mark.parametrize("mode", ALL_MODES)
def test_returns_one_point_per_pointid_for_every_mode(mode):
    pointids = ["a", "b", "c", "d", "e", "f", "g"]
    points = generate_2d_points(pointids, mode, sigma=0.2, random_state=random.Random(0))

    assert set(points.keys()) == set(pointids)
    for x, y in points.values():
        assert isinstance(x, float)
        assert isinstance(y, float)


def test_unknown_mode_raises_value_error():
    with pytest.raises(ValueError, match="not known"):
        generate_2d_points(["a"], "not-a-real-mode", sigma=0.2, random_state=random.Random(0))


def test_same_seed_gives_same_points():
    pointids = list(range(10))
    points_a = generate_2d_points(pointids, "eucl2", sigma=0.3, random_state=random.Random(42))
    points_b = generate_2d_points(pointids, "eucl2", sigma=0.3, random_state=random.Random(42))

    assert points_a == points_b


def test_different_seed_gives_different_points():
    pointids = list(range(10))
    points_a = generate_2d_points(pointids, "eucl2", sigma=0.3, random_state=random.Random(1))
    points_b = generate_2d_points(pointids, "eucl2", sigma=0.3, random_state=random.Random(2))

    assert points_a != points_b


def test_eucl1_rejection_sampling_stays_within_bounds():
    pointids = list(range(30))
    # a large sigma stresses the eucl1 rejection-sampling loop the most,
    # since more draws land outside [-1, 1] and get retried.
    points = generate_2d_points(pointids, "eucl1", sigma=5.0, random_state=random.Random(7))

    for x, y in points.values():
        assert -1 <= x <= 1
        assert -1 <= y <= 1
