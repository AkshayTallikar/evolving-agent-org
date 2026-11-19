import pytest

from evorg.stats import bootstrap_ci, paired_bootstrap_ci

VALUES = [0.42, 0.55, 0.61, 0.38, 0.5, 0.47, 0.66, 0.44, 0.52, 0.58]


def test_interval_is_deterministic_for_a_seed():
    assert bootstrap_ci(VALUES, seed=7) == bootstrap_ci(VALUES, seed=7)


def test_interval_contains_the_mean_and_is_ordered():
    low, high = bootstrap_ci(VALUES)
    mean = sum(VALUES) / len(VALUES)
    assert low <= mean <= high
    assert low < high


def test_interval_narrows_with_more_data():
    wide = bootstrap_ci(VALUES)
    narrow = bootstrap_ci(VALUES * 10)
    assert narrow[1] - narrow[0] < wide[1] - wide[0]


def test_constant_input_gives_zero_width_interval():
    low, high = bootstrap_ci([0.3] * 5)
    assert low == high
    assert low == pytest.approx(0.3)


def test_single_value_gives_zero_width_interval():
    low, high = bootstrap_ci([0.25])
    assert low == high == pytest.approx(0.25)


def test_paired_difference_of_identical_arms_is_zero_width_at_zero():
    low, high = paired_bootstrap_ci(VALUES, VALUES)
    assert low == high == pytest.approx(0.0)


def test_paired_interval_excludes_zero_for_a_clear_shift():
    base = [0.4, 0.5, 0.45, 0.42, 0.48, 0.51, 0.39, 0.44]
    shifted = [x + 0.2 for x in base]
    low, high = paired_bootstrap_ci(shifted, base)
    assert low > 0
    assert low <= high


def test_paired_interval_is_deterministic_for_a_seed():
    a, b = VALUES, list(reversed(VALUES))
    assert paired_bootstrap_ci(a, b, seed=3) == paired_bootstrap_ci(a, b, seed=3)


def test_paired_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0, 2.0], [1.0])


def test_empty_input_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_empty_paired_input_raises():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([], [])


@pytest.mark.parametrize("level", [0.0, 1.0, 1.5])
def test_level_outside_unit_interval_raises(level):
    with pytest.raises(ValueError):
        bootstrap_ci(VALUES, level=level)
