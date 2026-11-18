import sys
from types import SimpleNamespace

import pytest

from evorg.llm import AnthropicBackend, LLMRequest, SimulatedBackend, make_backend
from evorg.llm.base import MODEL_IDS, estimate_cost


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response):
        self.messages = FakeMessages(response)


def _response(blocks, input_tokens=1000, output_tokens=2000, stop_reason="end_turn"):
    return SimpleNamespace(
        content=blocks,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason=stop_reason,
    )


def _request(model="small"):
    return LLMRequest("coder", model, "Be careful.", "Write add_one.", task_id="t1")


def test_complete_joins_text_blocks_and_ignores_other_blocks():
    blocks = [
        SimpleNamespace(type="text", text="Hello "),
        SimpleNamespace(type="thinking", text="secret reasoning"),
        SimpleNamespace(type="text", text="world"),
    ]
    client = FakeClient(_response(blocks))
    result = AnthropicBackend(client=client).complete(_request())
    assert result.text == "Hello world"


def test_complete_reports_tokens_and_cost():
    client = FakeClient(_response([SimpleNamespace(type="text", text="ok")]))
    result = AnthropicBackend(client=client).complete(_request("small"))
    assert result.input_tokens == 1000
    assert result.output_tokens == 2000
    assert result.cost_usd == pytest.approx(estimate_cost("small", 1000, 2000))
    assert result.cost_usd == pytest.approx(0.0011)


def test_complete_builds_the_messages_call():
    client = FakeClient(_response([SimpleNamespace(type="text", text="ok")]))
    AnthropicBackend(client=client, effort="high").complete(_request("large"))
    (call,) = client.messages.calls
    assert call == {
        "model": MODEL_IDS["large"],
        "max_tokens": 4000,
        "system": "Be careful.",
        "messages": [{"role": "user", "content": "Write add_one."}],
        "output_config": {"effort": "high"},
    }


def test_default_effort_is_low():
    client = FakeClient(_response([SimpleNamespace(type="text", text="ok")]))
    AnthropicBackend(client=client).complete(_request())
    assert client.messages.calls[0]["output_config"] == {"effort": "low"}


def test_refusal_raises_runtime_error():
    client = FakeClient(_response([], stop_reason="refusal"))
    with pytest.raises(RuntimeError):
        AnthropicBackend(client=client).complete(_request())


def test_unknown_tier_is_rejected_before_calling_the_client():
    client = FakeClient(_response([]))
    with pytest.raises(ValueError):
        AnthropicBackend(client=client).complete(_request("medium"))
    assert client.messages.calls == []


def test_missing_package_gives_clear_error(monkeypatch):
    # A None entry in sys.modules makes "import anthropic" raise ImportError.
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(RuntimeError, match=r"pip install evorg\[anthropic\]"):
        AnthropicBackend()


def test_name_is_anthropic():
    assert AnthropicBackend(client=FakeClient(_response([]))).name == "anthropic"


def test_make_backend_builds_sim_and_anthropic():
    sim = make_backend("sim", task_index={}, seed=3)
    assert isinstance(sim, SimulatedBackend)
    assert sim.seed == 3
    client = FakeClient(_response([]))
    assert isinstance(make_backend("anthropic", client=client), AnthropicBackend)


def test_make_backend_rejects_unknown_names():
    with pytest.raises(ValueError):
        make_backend("openai")
