"""Backend that calls the Anthropic Messages API. Only used when explicitly selected."""

from __future__ import annotations

from typing import Any

from evorg.llm.base import MODEL_IDS, LLMRequest, LLMResponse, estimate_cost

MAX_TOKENS = 4000


class AnthropicBackend:
    """Calls ``client.messages.create`` with the tier mapped through ``MODEL_IDS``.

    If ``client`` is None, a real ``anthropic.Anthropic()`` client is built. The package is
    imported only at that point, so offline use and tests never need it installed.
    """

    name = "anthropic"

    def __init__(self, client: Any = None, effort: str = "low") -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "the anthropic backend needs the anthropic package; "
                    "install it with: pip install evorg[anthropic]"
                ) from exc
            client = anthropic.Anthropic()
        self.client = client
        self.effort = effort

    def complete(self, request: LLMRequest) -> LLMResponse:
        if request.model not in MODEL_IDS:
            raise ValueError(f"unknown model tier {request.model!r}")
        response = self.client.messages.create(
            model=MODEL_IDS[request.model],
            max_tokens=MAX_TOKENS,
            system=request.system,
            messages=[{"role": "user", "content": request.prompt}],
            output_config={"effort": self.effort},
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"the model refused the {request.role_kind} request")
        text = "".join(block.text for block in response.content if block.type == "text")
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = estimate_cost(request.model, input_tokens, output_tokens)
        return LLMResponse(
            text=text, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost
        )
