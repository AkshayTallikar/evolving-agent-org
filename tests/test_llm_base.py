import pytest

from evorg.llm.base import (
    MODEL_IDS,
    PRICING_PER_MTOK,
    LLMBackend,
    LLMRequest,
    count_tokens_approx,
    estimate_cost,
)


def test_model_ids_and_pricing_cover_both_tiers():
    assert set(MODEL_IDS) == {"small", "large"} == set(PRICING_PER_MTOK)
    assert MODEL_IDS["small"] == "claude-haiku-5-5"


def test_estimate_cost_is_per_million_tokens():
    # small: $0.10 in, $0.50 out per million tokens
    assert estimate_cost("small", 1_000_000, 0) == pytest.approx(0.10)
    assert estimate_cost("small", 0, 1_000_000) == pytest.approx(0.50)
    assert estimate_cost("small", 2_000, 500) == pytest.approx(2_000 / 1e6 * 0.10 + 500 / 1e6 * 0.5)


def test_estimate_cost_large_tier_and_zero_tokens():
    assert estimate_cost("large", 1_000_000, 1_000_000) == pytest.approx(24.0)
    assert estimate_cost("large", 0, 0) == 0.0


def test_estimate_cost_rejects_unknown_tier():
    with pytest.raises(ValueError):
        estimate_cost("medium", 10, 10)


def test_count_tokens_approx_is_quarter_length_with_floor_of_one():
    assert count_tokens_approx("") == 1
    assert count_tokens_approx("abc") == 1
    assert count_tokens_approx("a" * 400) == 100


def test_request_is_hashable_and_metadata_is_ignored_for_equality():
    a = LLMRequest("coder", "small", "sys", "prompt", task_id="t1", metadata={"x": 1})
    b = LLMRequest("coder", "small", "sys", "prompt", task_id="t1", metadata={"x": 2})
    assert a == b
    assert hash(a) == hash(b)


def test_protocol_recognises_objects_with_name_and_complete():
    class Dummy:
        name = "dummy"

        def complete(self, request):
            raise NotImplementedError

    assert isinstance(Dummy(), LLMBackend)
    assert not isinstance(object(), LLMBackend)
