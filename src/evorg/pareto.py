"""Pareto ranking of genomes: maximise pass rate, minimise mean cost per task.

The ranking is NSGA-II style: non-dominated fronts first, then crowding distance inside each
front. Every output is a function of the set of points, not of their input order. Ties are
broken by ``key`` (the genome id), so the same points always give the same answer. Design
section 4.10 describes the contract.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from operator import attrgetter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evorg.evolution import Evaluation

INF = float("inf")


@dataclass(frozen=True)
class Point:
    """One genome's objective values: pass rate (maximised) and cost in USD (minimised)."""

    key: str  # genome id
    pass_rate: float
    cost: float


def dominates(a: Point, b: Point) -> bool:
    """Return True when ``a`` is at least as good as ``b`` on both objectives and better on one.

    Two points with identical objective values do not dominate each other.
    """
    at_least_as_good = a.pass_rate >= b.pass_rate and a.cost <= b.cost
    strictly_better = a.pass_rate > b.pass_rate or a.cost < b.cost
    return at_least_as_good and strictly_better


def non_dominated_fronts(points: Sequence[Point]) -> list[list[Point]]:
    """Split ``points`` into Pareto fronts; front 0 is the set no other point dominates.

    A point's front index is one more than the largest front index among the points that
    dominate it. Each front is ordered by cost ascending, then key.

    Raises:
        ValueError: if two points share a key.
    """
    remaining = _canonical(points)
    fronts: list[list[Point]] = []
    while remaining:
        front = [p for p in remaining if not any(dominates(q, p) for q in remaining)]
        taken = {p.key for p in front}
        remaining = [p for p in remaining if p.key not in taken]
        fronts.append(front)
    return fronts


def crowding_distance(front: Sequence[Point]) -> dict[str, float]:
    """Crowding distance of each point in ``front``, keyed by point key.

    Each objective is normalised by the front's range for that objective. Boundary points
    (lowest and highest value) get infinity. An objective whose range is 0 contributes nothing.
    Fronts of one or two points give infinity to every point.

    Raises:
        ValueError: if two points share a key.
    """
    _check_unique_keys(front)
    if len(front) <= 2:
        return {p.key: INF for p in front}
    distance = {p.key: 0.0 for p in front}
    for objective in (attrgetter("pass_rate"), attrgetter("cost")):
        _add_objective_distance(front, objective, distance)
    return distance


def pareto_rank(points: Sequence[Point]) -> dict[str, tuple[int, float]]:
    """Map each key to ``(front index, crowding distance)``. Front 0 is the Pareto frontier.

    Raises:
        ValueError: if two points share a key.
    """
    ranks: dict[str, tuple[int, float]] = {}
    for index, front in enumerate(non_dominated_fronts(points)):
        for key, distance in crowding_distance(front).items():
            ranks[key] = (index, distance)
    return ranks


def pareto_sort_key(rank: tuple[int, float], key: str) -> tuple[int, float, str]:
    """Sort key that puts the best points first: lower front, then higher crowding, then key.

    Crowding is negated so that a plain ascending sort puts the largest distance first.
    """
    front, crowding = rank
    return (front, -crowding, key)


def frontier(points: Sequence[Point]) -> list[Point]:
    """The non-dominated set (front 0), sorted by cost ascending, then key."""
    fronts = non_dominated_fronts(points)
    return sorted(fronts[0], key=_cost_then_key) if fronts else []


def points_from_evaluations(evaluations: Mapping[str, Evaluation]) -> list[Point]:
    """Convert evaluations to points, using the mapping key as the genome key.

    Pass rate maps to ``pass_rate`` and ``mean_cost_usd`` maps to ``cost``. Points are returned
    in key order.
    """
    return [
        Point(key=key, pass_rate=evaluations[key].pass_rate, cost=evaluations[key].mean_cost_usd)
        for key in sorted(evaluations)
    ]


def _cost_then_key(point: Point) -> tuple[float, str]:
    return (point.cost, point.key)


def _canonical(points: Sequence[Point]) -> list[Point]:
    """Points in cost-then-key order, so results do not depend on the input order."""
    _check_unique_keys(points)
    return sorted(points, key=_cost_then_key)


def _check_unique_keys(points: Sequence[Point]) -> None:
    seen: set[str] = set()
    for point in points:
        if point.key in seen:
            raise ValueError(f"duplicate genome key {point.key!r}")
        seen.add(point.key)


def _add_objective_distance(
    front: Sequence[Point],
    objective: Callable[[Point], float],
    distance: dict[str, float],
) -> None:
    """Add one objective's normalised neighbour gaps to ``distance`` in place."""
    ordered = sorted(front, key=lambda p: (objective(p), p.key))
    span = objective(ordered[-1]) - objective(ordered[0])
    if span == 0:
        return
    distance[ordered[0].key] = INF
    distance[ordered[-1].key] = INF
    for before, point, after in zip(ordered, ordered[1:], ordered[2:], strict=False):
        distance[point.key] += (objective(after) - objective(before)) / span
