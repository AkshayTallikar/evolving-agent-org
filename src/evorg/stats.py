"""Percentile bootstrap intervals, sign-flip permutation tests and Holm adjustment over seeds.

Standard library only.
"""

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


EXACT_MAX_N = 16  # enumerate all 2**n sign patterns up to this many pairs


def paired_permutation_p(
    a: Sequence[float], b: Sequence[float], *, reps: int = 10000, seed: int = 0
) -> float:
    """Two-sided sign-flip permutation p-value for mean(a - b) = 0.

    Under the null, each paired difference is equally likely to have either sign. The p-value
    is the share of sign patterns whose mean difference is at least as far from zero as the
    observed one. With at most ``EXACT_MAX_N`` pairs every pattern is enumerated, so the answer
    is exact. Otherwise ``reps`` patterns are drawn with a ``random.Random`` seeded with
    ``seed``, and the count gets the usual +1 correction, so p is never 0. Identical arms, or
    differences that are all zero, give p = 1.
    """
    if len(a) != len(b):
        raise ValueError(f"paired samples differ in length: {len(a)} and {len(b)}")
    if len(a) == 0:
        raise ValueError("paired_permutation_p needs at least one pair")
    if reps < 1:
        raise ValueError("reps must be at least 1")
    diffs = [float(x) - float(y) for x, y in zip(a, b, strict=True)]
    scale = max(abs(d) for d in diffs)
    if scale == 0:
        return 1.0
    tol = 1e-9 * scale  # sums of the same magnitudes can differ in the last bits
    n = len(diffs)
    observed = abs(math.fsum(diffs)) / n

    if n <= EXACT_MAX_N:
        total = 0
        for pattern in range(1 << n):
            signed = [-d if (pattern >> i) & 1 else d for i, d in enumerate(diffs)]
            if abs(math.fsum(signed)) / n >= observed - tol:
                total += 1
        return total / (1 << n)

    rng = random.Random(seed)
    extreme = 0
    for _ in range(reps):
        signed = [d if rng.random() < 0.5 else -d for d in diffs]
        if abs(math.fsum(signed)) / n >= observed - tol:
            extreme += 1
    return (extreme + 1) / (reps + 1)


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the input order.

    Sort the p-values ascending; the j-th smallest (1-based) is multiplied by m - j + 1, where
    m is the number of p-values, then a running maximum keeps the adjusted values monotone in
    the sorted order. Every value is capped at 1. An adjusted value is never below its raw one.
    """
    m = len(pvalues)
    for p in pvalues:
        if not 0 <= p <= 1:
            raise ValueError(f"p-value must be between 0 and 1, got {p}")
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted
