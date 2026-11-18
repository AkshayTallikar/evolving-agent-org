"""Backend-neutral types for LLM calls: requests, responses, pricing and token counts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

MODEL_IDS: dict[str, str] = {"small": "claude-haiku-5-5", "large": "claude-opus-5-5"}

# (input, output) price in USD per million tokens, keyed by model tier.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "small": (0.10, 0.50),
    "large": (4.00, 20.00),
}


@dataclass(frozen=True)
class LLMRequest:
    """One call to a backend. ``model`` is a tier name ("small" or "large")."""

    role_kind: str
    model: str
    system: str
    prompt: str
    task_id: str | None = None
    attempt: int = 0
    metadata: dict = field(default_factory=dict, hash=False, compare=False)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@runtime_checkable
class LLMBackend(Protocol):
    """Anything with a ``name`` and a ``complete`` method can serve as a backend."""

    name: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost of a call on tier ``model``. Raises ValueError for an unknown tier."""
    if model not in PRICING_PER_MTOK:
        known = sorted(PRICING_PER_MTOK)
        raise ValueError(f"unknown model tier {model!r}; expected one of {known}")
    price_in, price_out = PRICING_PER_MTOK[model]
    return input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out


def count_tokens_approx(text: str) -> int:
    """Rough token count: one token per four characters, never less than one."""
    return max(1, len(text) // 4)
