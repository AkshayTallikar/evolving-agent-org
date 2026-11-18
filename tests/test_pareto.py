import random

import pytest

from evorg.evolution import Evaluation
from evorg.pareto import (
    Point,
    crowding_distance,
    dominates,
    frontier,
    non_dominated_fronts,
    pareto_rank,
    pareto_sort_key,
    points_from_evaluations,
)

INF = float("inf")


def _p(key: str, pass_rate: float, cost: float) -> Point:
    return Point(key=key, pass_rate=pass_rate, cost=cost)


def _keys(front: list[Point]) -> list[str]:
    return [p.key for p in front]


def _random_points(rng: random.Random, n: int) -> list[Point]:
    # Coarse grids force many ties, including exact duplicates.
    return [_p(f"g{i:02d}", round(rng.random(), 1), round(rng.random() * 4) / 2) for i in range(n)]


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (_p("a", 0.9, 1.0), _p("b", 0.5, 2.0), True),  # better on both
        (_p("a", 0.9, 1.0), _p("b", 0.5, 1.0), True),  # better rate, equal cost
        (_p("a", 0.5, 0.5), _p("b", 0.5, 1.0), True),  # equal rate, lower cost
        (_p("a", 0.5, 1.0), _p("b", 0.5, 1.0), False),  # identical values: tie
        (_p("a", 0.9, 2.0), _p("b", 0.5, 1.0), False),  # trade-off: neither dominates
        (_p("b", 0.5, 1.0), _p("a", 0.9, 2.0), False),  # trade-off, reversed
        (_p("a", 0.5, 2.0), _p("b", 0.9, 1.0), False),  # worse on both
        (_p("a", 0.5, 2.0), _p("b", 0.5, 1.0), False),  # equal rate, higher cost
    ],
)
def test_dominates_truth_table(a: Point, b: Point, expected: bool) -> None:
    assert dominates(a, b) is expected


def test_identical_points_share_a_front_and_do_not_dominate() -> None:
    twins = [_p("x", 0.7, 1.0), _p("y", 0.7, 1.0)]
    assert not dominates(twins[0], twins[1])
    assert not dominates(twins[1], twins[0])
    fronts = non_dominated_fronts(twins)
    assert [_keys(front) for front in fronts] == [["x", "y"]]


def test_three_known_fronts() -> None:
    points = [
        _p("a", 0.9, 3.0),  # front 0: most accurate, expensive
        _p("b", 0.7, 1.0),  # front 0: cheap
        _p("c", 0.6, 1.5),  # dominated by b only
        _p("d", 0.8, 3.5),  # dominated by a only
        _p("e", 0.5, 2.0),  # dominated by b and c
    ]
    fronts = non_dominated_fronts(points)
    assert [_keys(front) for front in fronts] == [["b", "a"], ["c", "d"], ["e"]]


def test_crowding_distance_hand_computed() -> None:
    front = [_p("A", 0.5, 1.0), _p("B", 0.6, 2.0), _p("C", 0.8, 3.0), _p("D", 0.9, 5.0)]
    distance = crowding_distance(front)
    # pass_rate range 0.4, cost range 4.
    # B: pass (0.8 - 0.5) / 0.4 = 0.75, cost (3 - 1) / 4 = 0.5 -> 1.25
    # C: pass (0.9 - 0.6) / 0.4 = 0.75, cost (5 - 2) / 4 = 0.75 -> 1.5
    assert distance["A"] == INF
    assert distance["D"] == INF
    assert distance["B"] == pytest.approx(1.25)
    assert distance["C"] == pytest.approx(1.5)


def test_zero_range_objective_contributes_nothing() -> None:
    same = [_p("p", 0.5, 1.0), _p("q", 0.5, 1.0), _p("r", 0.5, 1.0)]
    assert crowding_distance(same) == {"p": 0.0, "q": 0.0, "r": 0.0}

    # Same cost everywhere: only pass_rate contributes. Boundary points get inf from pass_rate.
    flat_cost = [_p("a", 0.5, 1.0), _p("b", 0.7, 1.0), _p("c", 0.9, 1.0)]
    distance = crowding_distance(flat_cost)
    assert distance["a"] == INF
    assert distance["c"] == INF
    assert distance["b"] == pytest.approx(1.0)


def test_boundary_points_are_infinite_in_pareto_rank() -> None:
    front = [_p(str(i), 0.2 * i, float(i)) for i in range(1, 6)]
    ranks = pareto_rank(front)
    assert all(front_index == 0 for front_index, _ in ranks.values())
    assert ranks["1"][1] == INF
    assert ranks["5"][1] == INF
    assert all(0 < ranks[k][1] < INF for k in ("2", "3", "4"))


@pytest.mark.parametrize("size", [1, 2])
def test_small_fronts_are_infinite(size: int) -> None:
    front = [_p(f"s{i}", 0.1 * i, float(i)) for i in range(size)]
    assert crowding_distance(front) == {p.key: INF for p in front}


def test_sort_key_orders_by_front_then_crowding_then_key() -> None:
    ranks = {
        "a": (1, 0.5),
        "b": (0, 0.1),
        "c": (0, INF),
        "d": (1, 0.5),
        "e": (0, 0.1),
    }
    ordered = sorted(ranks, key=lambda k: pareto_sort_key(ranks[k], k))
    assert ordered == ["c", "b", "e", "a", "d"]


def test_frontier_is_front_zero_sorted_by_cost_then_key() -> None:
    points = [
        _p("z", 0.9, 3.0),
        _p("m", 0.7, 1.0),
        _p("a", 0.7, 1.0),  # exact duplicate of m: both stay on the frontier
        _p("k", 0.2, 0.5),
        _p("bad", 0.1, 4.0),  # dominated
    ]
    result = frontier(points)
    assert [(p.key, p.cost) for p in result] == [("k", 0.5), ("a", 1.0), ("m", 1.0), ("z", 3.0)]


def test_frontier_of_empty_input_is_empty() -> None:
    assert frontier([]) == []
    assert non_dominated_fronts([]) == []
    assert pareto_rank([]) == {}


def test_output_is_invariant_to_input_order() -> None:
    rng = random.Random(7)
    base = _random_points(rng, 12)
    expected_fronts = [_keys(f) for f in non_dominated_fronts(base)]
    expected_ranks = pareto_rank(base)
    expected_frontier = _keys(frontier(base))

    shuffler = random.Random(1234)
    for _ in range(50):
        shuffled = list(base)
        shuffler.shuffle(shuffled)
        assert [_keys(f) for f in non_dominated_fronts(shuffled)] == expected_fronts
        assert pareto_rank(shuffled) == expected_ranks
        assert _keys(frontier(shuffled)) == expected_frontier


def test_random_sets_satisfy_front_structure() -> None:
    rng = random.Random(99)
    for _ in range(200):
        points = _random_points(rng, rng.randint(1, 25))
        fronts = non_dominated_fronts(points)

        # Every point appears in exactly one front.
        assert sorted(p.key for front in fronts for p in front) == sorted(p.key for p in points)

        for k, front in enumerate(fronts):
            later = [q for later_front in fronts[k:] for q in later_front]
            for p in front:
                # Nothing in front k or later dominates a point in front k.
                assert not any(dominates(q, p) for q in later)
                # Every point in front k > 0 is dominated by some point in front k - 1.
                if k > 0:
                    assert any(dominates(q, p) for q in fronts[k - 1])


def test_duplicate_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate genome key 'g'"):
        non_dominated_fronts([_p("g", 0.1, 1.0), _p("g", 0.2, 2.0)])


def test_points_from_evaluations_maps_objectives_in_key_order() -> None:
    def ev(genome_id: str, pass_rate: float, cost: float) -> Evaluation:
        return Evaluation(
            genome_id=genome_id,
            pass_rate=pass_rate,
            mean_cost_usd=cost,
            fitness=pass_rate - cost,
            results=(),
        )

    evaluations = {"g2": ev("g2", 0.25, 0.02), "g1": ev("g1", 0.75, 0.05)}
    assert points_from_evaluations(evaluations) == [
        Point(key="g1", pass_rate=0.75, cost=0.05),
        Point(key="g2", pass_rate=0.25, cost=0.02),
    ]
    assert points_from_evaluations({}) == []
