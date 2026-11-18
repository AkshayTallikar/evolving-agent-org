"""LLM backends for evorg: a protocol, a simulated backend and an Anthropic backend."""

from typing import Any

from evorg.llm.anthropic_backend import AnthropicBackend
from evorg.llm.base import (
    MODEL_IDS,
    PRICING_PER_MTOK,
    LLMBackend,
    LLMRequest,
    LLMResponse,
    count_tokens_approx,
    estimate_cost,
)
from evorg.llm.simulated import SimulatedBackend

__all__ = [
    "MODEL_IDS",
    "PRICING_PER_MTOK",
    "AnthropicBackend",
    "LLMBackend",
    "LLMRequest",
    "LLMResponse",
    "SimulatedBackend",
    "count_tokens_approx",
    "estimate_cost",
    "make_backend",
]


def make_backend(name: str, **kw: Any) -> LLMBackend:
    """Build a backend by name: "sim" or "anthropic". Keyword arguments go to its constructor."""
    if name == "sim":
        return SimulatedBackend(**kw)
    if name == "anthropic":
        return AnthropicBackend(**kw)
    raise ValueError(f"unknown backend {name!r}; expected 'sim' or 'anthropic'")
