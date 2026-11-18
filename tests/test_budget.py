"""Spending guard and cost estimate: fake inner backends only, no network."""

from __future__ import annotations

import math
import random
import threading
import time

import pytest

from evorg.genome import seed_genome
from evorg.llm import (
    BudgetedBackend,
    BudgetExceeded,
    CountingBackend,
    LLMRequest,
    LLMResponse,
    estimate_cost,
)
from evorg.llm.anthropic_backend import MAX_TOKENS
from evorg.llm.budget import UsageTally, estimate_upper_bound, worst_case_cost
from evorg.mutation import Architect


class FakeInner:
    """A backend that returns a fixed cost, or raises, and counts how often it was called."""

    name = "fake"

    def __init__(self, cost: float = 0.01, error: BaseException | None = None) -> None:
        self.cost = cost
        self.error = error
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, request: LLMRequest) -> LLMResponse:
        with self._lock:
            self.calls += 1
        if self.error is not None:
            raise self.error
        time.sleep(0.001)
        return LLMResponse(text="ok", input_tokens=10, output_tokens=5, cost_usd=self.cost)


def _request(model: str = "small", prompt: str = "write add_one") -> LLMRequest:
    return LLMRequest("coder", model, "Be careful.", prompt, task_id="t1")


def test_cap_blocks_a_call_whose_worst_case_would_cross_it():
    inner = FakeInner()
    budget = BudgetedBackend(inner, limit_usd=0.001)  # worst case for one call is about 0.002
    with pytest.raises(BudgetExceeded):
        budget.complete(_request())
    assert inner.calls == 0
    assert budget.spent_usd == 0.0
    assert budget.calls == 0


def test_blocked_error_reports_spent_and_limit():
    budget = BudgetedBackend(FakeInner(cost=0.5), limit_usd=0.5001)
    budget.complete(_request())  # spends 0.5; the next worst case does not fit
    with pytest.raises(BudgetExceeded) as exc:
        budget.complete(_request())
    assert exc.value.spent_usd == pytest.approx(0.5)
    assert exc.value.limit_usd == pytest.approx(0.5001)


def test_spend_accumulates_actual_costs():
    inner = FakeInner(cost=0.01)
    budget = BudgetedBackend(inner, limit_usd=1.0)
    for _ in range(3):
        budget.complete(_request())
    assert inner.calls == 3
    assert budget.spent_usd == pytest.approx(0.03)
    assert budget.calls == 3
    assert budget.limit_usd == 1.0
    assert budget.name == "fake"


def test_a_failed_call_is_charged_its_worst_case():
    # A refused or timed-out request may still be billed, so the cap assumes it was.
    worst = worst_case_cost(_request())
    inner = FakeInner(error=RuntimeError("network down"))
    budget = BudgetedBackend(inner, limit_usd=worst * 1.5)  # room for one worst case
    with pytest.raises(RuntimeError, match="network down"):
        budget.complete(_request())
    assert budget.spent_usd == pytest.approx(worst)
    assert budget.calls == 0
    inner.error = None
    with pytest.raises(BudgetExceeded):
        budget.complete(_request())  # a second worst case would cross the cap
    assert inner.calls == 1


def test_concurrent_calls_never_exceed_the_limit():
    limit = 0.1
    inner = FakeInner(cost=0.001)  # real costs stay below the worst case reserved per call
    budget = BudgetedBackend(inner, limit_usd=limit)
    refused = 0
    refused_lock = threading.Lock()

    def worker() -> None:
        nonlocal refused
        for _ in range(20):
            try:
                budget.complete(_request())
            except BudgetExceeded:
                with refused_lock:
                    refused += 1

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert budget.spent_usd <= limit + 1e-12
    assert budget.spent_usd == pytest.approx(inner.calls * 0.001)
    assert refused > 0
    assert inner.calls + refused == 160


def test_non_positive_limit_is_rejected():
    with pytest.raises(ValueError):
        BudgetedBackend(FakeInner(), limit_usd=0)


def test_worst_case_uses_safety_factor_and_max_tokens():
    request = _request(prompt="x" * 400)
    tokens_in = math.ceil((len("Be careful.") + 400) // 4 * 1.5)
    assert worst_case_cost(request) == pytest.approx(estimate_cost("small", tokens_in, MAX_TOKENS))


def test_architect_calls_count_against_the_same_budget():
    inner = FakeInner(cost=0.02)
    budget = BudgetedBackend(inner, limit_usd=10.0)
    architect = Architect(budget, random.Random(0))
    architect.propose(seed_genome(), "", n=2)
    assert architect.stats["cost_usd"] == pytest.approx(0.02)
    assert budget.spent_usd == pytest.approx(architect.stats["cost_usd"])


def test_counting_backend_records_calls_and_tokens_per_tier():
    tally = UsageTally()
    counting = CountingBackend(FakeInner(), tally)
    counting.complete(_request("small"))
    counting.complete(_request("large"))
    counting.complete(_request("large"))
    assert tally.calls == {"small": 1, "large": 2}
    assert tally.input_tokens == {"small": 10, "large": 20}
    assert tally.output_tokens == {"small": 5, "large": 10}


def test_upper_bound_prices_worst_case_output_and_scaled_input():
    tally = UsageTally()
    tally.add("small", 100, 50)
    bound = estimate_upper_bound(tally)
    assert bound.calls == {"small": 1}
    assert bound.total_usd == pytest.approx(estimate_cost("small", 150, MAX_TOKENS))


def test_large_share_splits_calls_between_tiers():
    tally = UsageTally()
    tally.add("small", 100, 10)
    tally.add("small", 100, 10)
    bound = estimate_upper_bound(tally, large_share=0.5)
    assert bound.calls == {"small": 1, "large": 1}
    assert bound.total_calls == 2
    with pytest.raises(ValueError):
        estimate_upper_bound(tally, large_share=1.5)
