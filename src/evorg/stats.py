"""Percentile bootstrap confidence intervals for means over seeds (standard library only)."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence


def _check_level(level: float, reps: int) -> None:
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    if reps < 1:
        raise ValueError("reps must be at least 1")


def bootstrap_ci(
    values: Sequence[float], *, reps: int = 2000, level: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``values``.

    Resamples ``values`` with replacement ``reps`` times, using a ``random.Random`` seeded
    with ``seed``, and returns the ``level`` central range of the resampled means. The same
    inputs always give the same interval. Constant input gives a zero-width interval.
    """
    _check_level(level, reps)
    if len(values) == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    data = [float(v) for v in values]
    rng = random.Random(seed)
    n = len(data)
    means = sorted(statistics.fmean(rng.choices(data, k=n)) for _ in range(reps))
    tail = (1 - level) / 2
    low = means[math.floor(tail * reps)]
    high = means[min(reps - 1, math.ceil((1 - tail) * reps) - 1)]
    return low, high


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    reps: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap interval for the mean of the paired differences ``a[i] - b[i]``.

    ``a[i]`` and ``b[i]`` must come from the same seed. An interval that excludes zero is
    evidence of a difference between the two arms.
    """
    if len(a) != len(b):
        raise ValueError(f"paired samples differ in length: {len(a)} and {len(b)}")
    return bootstrap_ci(
        [x - y for x, y in zip(a, b, strict=True)], reps=reps, level=level, seed=seed
    )
