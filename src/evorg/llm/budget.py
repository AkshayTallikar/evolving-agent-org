"""Spending guard: a hard cap on what a backend may spend, and a counter for cost estimates.

``BudgetedBackend`` reserves the worst-case cost of each call before making it, so a call
that could push spending over the cap is refused before any money is spent. Reservations
are taken under a lock, so calls from several threads cannot overshoot together.

``CountingBackend`` records calls and tokens per model tier without changing any answer.
``estimate_upper_bound`` turns those counts into a dollar bound for a real run.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

from evorg.llm.anthropic_backend import MAX_TOKENS
from evorg.llm.base import (
    LLMBackend,
    LLMRequest,
    LLMResponse,
    count_tokens_approx,
    estimate_cost,
)

# Real tokenizers differ from the four-characters-per-token approximation, so worst-case
# input is taken as this multiple of the approximate count.
SAFETY_FACTOR = 1.5


class BudgetExceeded(RuntimeError):
    """Raised before a call whose worst-case cost would take spending over the cap."""

    def __init__(self, spent_usd: float, limit_usd: float) -> None:
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd
        super().__init__(
            f"spending cap reached: ${spent_usd:.4f} spent of ${limit_usd:.2f} allowed"
        )


def worst_case_cost(request: LLMRequest) -> float:
    """Upper bound in USD for one request: safety-scaled input, output at ``MAX_TOKENS``."""
    approx_in = count_tokens_approx(request.system + request.prompt)
    input_tokens = math.ceil(approx_in * SAFETY_FACTOR)
    return estimate_cost(request.model, input_tokens, MAX_TOKENS)


class BudgetedBackend:
    """Wraps an LLM backend and refuses calls that would exceed ``limit_usd``.

    Before each call the worst-case cost is reserved; a call is refused with
    ``BudgetExceeded`` (and the inner backend is not touched) when the committed spend plus
    open reservations plus this reservation would exceed the limit. After a successful call
    the reservation is replaced by the actual ``cost_usd``. If the inner call raises, the
    worst case is charged: a refused or timed-out request may still have been billed, and
    the cap must never under-count.
    """

    def __init__(self, inner: LLMBackend, limit_usd: float) -> None:
        if not limit_usd > 0:
            raise ValueError(f"limit_usd must be positive, got {limit_usd}")
        self.inner = inner
        self.name = inner.name
        self._limit = float(limit_usd)
        self._spent = 0.0
        self._reserved = 0.0
        self._calls = 0
        self._lock = threading.Lock()

    @property
    def limit_usd(self) -> float:
        """The spending cap in USD."""
        return self._limit

    @property
    def spent_usd(self) -> float:
        """USD charged so far, from completed calls."""
        with self._lock:
            return self._spent

    @property
    def calls(self) -> int:
        """Number of calls that completed successfully."""
        with self._lock:
            return self._calls

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Reserve, call the inner backend, then settle to the actual cost."""
        worst = worst_case_cost(request)
        self._reserve(worst)
        try:
            response = self.inner.complete(request)
        except BaseException:
            self._settle(worst, worst, completed=False)
            raise
        self._settle(worst, float(response.cost_usd))
        return response

    def _reserve(self, worst: float) -> None:
        with self._lock:
            if self._spent + self._reserved + worst > self._limit:
                raise BudgetExceeded(self._spent, self._limit)
            self._reserved += worst

    def _settle(self, worst: float, actual: float, completed: bool = True) -> None:
        with self._lock:
            self._reserved -= worst
            self._spent += actual
            if completed:
                self._calls += 1


@dataclass
class UsageTally:
    """Calls and tokens per model tier, summed over every ``CountingBackend`` sharing it."""

    calls: dict[str, int] = field(default_factory=dict)
    input_tokens: dict[str, int] = field(default_factory=dict)
    output_tokens: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, tier: str, input_tokens: int, output_tokens: int) -> None:
        """Record one call on ``tier``."""
        with self._lock:
            self.calls[tier] = self.calls.get(tier, 0) + 1
            self.input_tokens[tier] = self.input_tokens.get(tier, 0) + input_tokens
            self.output_tokens[tier] = self.output_tokens.get(tier, 0) + output_tokens


class CountingBackend:
    """Forwards every call to ``inner`` and records its tokens in a ``UsageTally``."""

    def __init__(self, inner: LLMBackend, tally: UsageTally | None = None) -> None:
        self.inner = inner
        self.name = inner.name
        self.tally = tally if tally is not None else UsageTally()

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Call the inner backend and count the response's tokens under the request's tier."""
        response = self.inner.complete(request)
        self.tally.add(request.model, response.input_tokens, response.output_tokens)
        return response


@dataclass(frozen=True)
class CostBound:
    """Upper bound for a real run, per tier and in total."""

    calls: dict[str, int]
    usd_by_tier: dict[str, float]

    @property
    def total_calls(self) -> int:
        """Calls across all tiers."""
        return sum(self.calls.values())

    @property
    def total_usd(self) -> float:
        """Upper bound on the real run's spend in USD."""
        return sum(self.usd_by_tier.values())


def estimate_upper_bound(tally: UsageTally, large_share: float | None = None) -> CostBound:
    """Bound the cost of a real run from the calls a simulated run made.

    Every call is priced at ``MAX_TOKENS`` output and ``SAFETY_FACTOR`` times the simulated
    input tokens. The simulated tier mix is used as is, unless ``large_share`` (0 to 1) gives
    the fraction of calls that should run on the "large" tier; the calls then take the
    simulated average input size per call.
    """
    tiers = sorted(tally.calls)
    total = sum(tally.calls.values())
    if large_share is None:
        calls = {tier: tally.calls[tier] for tier in tiers}
        inputs = {tier: tally.input_tokens[tier] for tier in tiers}
    else:
        if not 0.0 <= large_share <= 1.0:
            raise ValueError(f"large_share must be between 0 and 1, got {large_share}")
        calls = _split_calls(total, large_share)
        avg_in = sum(tally.input_tokens.values()) / total if total else 0.0
        inputs = {tier: avg_in * calls[tier] for tier in calls}
    usd = {
        tier: estimate_cost(tier, math.ceil(inputs[tier] * SAFETY_FACTOR), calls[tier] * MAX_TOKENS)
        for tier in calls
    }
    return CostBound(calls=calls, usd_by_tier=usd)


def _split_calls(total: int, large_share: float) -> dict[str, int]:
    large = round(total * large_share)
    return {"small": total - large, "large": large}
