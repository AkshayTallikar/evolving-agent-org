import pytest

from evorg.stats import bootstrap_ci, holm_adjust, paired_bootstrap_ci, paired_permutation_p

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


def test_permutation_p_matches_hand_count_for_three_pairs():
    # diffs 1, 2, 3: observed mean 2. Sign patterns with |mean| >= 2 are +++ (2) and --- (-2),
    # out of 8 patterns, so p = 2/8.
    assert paired_permutation_p([2, 3, 4], [1, 1, 1]) == pytest.approx(0.25)


def test_permutation_p_with_two_equal_differences():
    # diffs 2, 2: patterns ++ (2) and -- (-2) reach the observed mean; +- and -+ give 0. p = 2/4.
    assert paired_permutation_p([2, 2], [0, 0]) == pytest.approx(0.5)


def test_permutation_p_ties_at_zero_mean_give_one():
    # diffs 1, -1: observed mean 0, every pattern has |mean| >= 0, so p = 1.
    assert paired_permutation_p([1, 0], [0, 1]) == pytest.approx(1.0)


def test_permutation_p_all_zero_differences_give_one():
    assert paired_permutation_p(VALUES, VALUES) == 1.0
    assert paired_permutation_p([0.0] * 30, [0.0] * 30) == 1.0


def test_permutation_p_is_symmetric_in_the_arms():
    a = [0.4, 0.5, 0.45, 0.42, 0.48, 0.51, 0.39, 0.44, 0.6, 0.55]
    b = [0.3, 0.52, 0.41, 0.4, 0.5, 0.45, 0.38, 0.47, 0.5, 0.6]
    assert paired_permutation_p(a, b) == pytest.approx(paired_permutation_p(b, a))


def test_permutation_p_detects_a_clear_shift_and_not_noise():
    base = [0.4, 0.5, 0.45, 0.42, 0.48, 0.51, 0.39, 0.44, 0.47, 0.5]
    shifted = [x + 0.2 for x in base]
    assert paired_permutation_p(shifted, base) < 0.01
    noisy = [0.5, 0.4, 0.6, 0.3, 0.55, 0.45, 0.35, 0.62, 0.42, 0.58]
    assert paired_permutation_p(noisy, base) > 0.2


def test_permutation_monte_carlo_is_seeded_and_never_zero():
    # 20 pairs goes past the exact limit. Every difference is +1, so only the all-positive
    # pattern reaches the observed mean; the +1 correction keeps p above zero.
    a = [1.0] * 20
    b = [0.0] * 20
    p = paired_permutation_p(a, b, reps=200, seed=4)
    assert p == paired_permutation_p(a, b, reps=200, seed=4)
    assert p >= 1 / 201


def test_permutation_rejects_bad_input():
    with pytest.raises(ValueError):
        paired_permutation_p([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        paired_permutation_p([], [])
    with pytest.raises(ValueError):
        paired_permutation_p([1.0], [0.0], reps=0)


def test_holm_adjusts_by_hand_and_keeps_input_order():
    # sorted: 0.005 (x4) -> 0.02, 0.01 (x1) -> 0.03, 0.03 (x3) -> 0.06, 0.04 (x2) -> 0.06
    got = holm_adjust([0.01, 0.04, 0.03, 0.005])
    assert got == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_is_capped_at_one_and_monotone():
    assert holm_adjust([0.6, 0.9]) == pytest.approx([1.0, 1.0])
    raw = [0.001, 0.2, 0.04, 0.5, 0.03]
    adjusted = holm_adjust(raw)
    assert all(a >= r for a, r in zip(adjusted, raw, strict=True))
    assert all(a <= 1.0 for a in adjusted)
    by_raw = sorted(zip(raw, adjusted, strict=True))
    assert [a for _, a in by_raw] == sorted(a for _, a in by_raw)


def test_holm_handles_ties_and_all_ones():
    assert holm_adjust([0.02, 0.02]) == pytest.approx([0.04, 0.04])
    assert holm_adjust([1.0, 1.0]) == [1.0, 1.0]
    assert holm_adjust([]) == []
    assert holm_adjust([0.5]) == [0.5]


def test_holm_rejects_p_values_outside_unit_interval():
    with pytest.raises(ValueError):
        holm_adjust([1.5])
