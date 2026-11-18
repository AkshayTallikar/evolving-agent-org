"""A deterministic stand-in for an LLM, so the whole system runs offline.

The simulated backend is not a random number generator with a prompt wrapper. It encodes a
small, hand-written capability model, so that the shape of an organization (which instructions
its roles carry, which tier they run on, whether there is a planner, reviewer or tester) has
measurable consequences that evolution can discover. The model is a caricature of real LLM
behaviour and says nothing about real models; see the honest limits in the design doc.

Capability model
----------------
``skill(request)`` is the sum of:

* ``FEATURE_WEIGHTS`` for each keyword found (lower-cased) in the system prompt plus the
  prompt. A keyword counts at most once, however often it appears.
* ``LARGE_MODEL_BONUS`` when the request runs on the "large" tier.
* ``PLAN_BONUS`` when the prompt contains a ``PLAN:`` block from the planner.
* ``FEEDBACK_BONUS`` when the prompt contains ``FEEDBACK:`` from the tester or reviewer.
* ``-INSTRUCTION_PENALTY`` for every system-prompt line starting with "- " beyond the first
  ``INSTRUCTION_FREE_LINES``. Long instruction lists dilute the prompt.

A coder then succeeds with ``p = sigmoid(skill + BASE_SKILL - difficulty)``. A reviewer catches
buggy code with ``sigmoid(skill - REVIEW_BIAS)`` and wrongly rejects correct code with a fixed
small probability.

All randomness comes from ``SimulatedBackend._unit``, a sha256 of the seed and the arguments,
so identical inputs give identical outputs on every machine.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from evorg.genes import INSTRUCTION_POOL
from evorg.llm.base import LLMRequest, LLMResponse, count_tokens_approx, estimate_cost

# Keyword -> effect on coder skill. Helpful phrases are positive, harmful ones negative,
# neutral ones zero. Keys are lower case; matching is on the lower-cased text.
FEATURE_WEIGHTS: dict[str, float] = {
    "edge case": 0.7,
    "empty input": 0.4,
    "step by step": 0.3,
    "mentally test": 0.5,
    "off-by-one": 0.5,
    "type hint": 0.05,
    "docstring": 0.0,
    "descriptive": 0.0,
    "be brief": -0.5,
    "skip": -0.6,
    "guess": -0.5,
    "speed over correctness": -0.7,
}

BASE_SKILL = 0.5
LARGE_MODEL_BONUS = 0.8
PLAN_BONUS = 0.4
FEEDBACK_BONUS = 0.6
INSTRUCTION_PENALTY = 0.05
INSTRUCTION_FREE_LINES = 6

# A reviewer catches buggy code with sigmoid(skill - REVIEW_BIAS); correct code is rejected
# with probability REVIEW_FALSE_ALARM.
REVIEW_BIAS = 0.5
REVIEW_FALSE_ALARM = 0.05

# Architect limits, mirroring the genome rules in design section 4.2.
ARCH_OP_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("add_instruction", 0.35),
    ("replace_instruction", 0.15),
    ("remove_instruction", 0.10),
    ("add_role", 0.15),
    ("remove_role", 0.05),
    ("change_model", 0.10),
    ("set_retries", 0.10),
)
ARCH_MAX_INSTRUCTIONS = 8
ARCH_MAX_ROLES = 6
ARCH_MAX_RETRIES = 4
ARCH_KIND_LIMITS: dict[str, int] = {"planner": 2, "tester": 1, "reviewer": 1}

_FENCE = re.compile(r"```python[ \t]*\n(.*?)```", re.DOTALL)
_REVIEW_HINT = "REVISE: check boundary conditions and edge cases"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _pick(options: Sequence[Any], u: float) -> Any:
    return options[min(int(u * len(options)), len(options) - 1)]


def _weighted_pick(items: Sequence[tuple[str, float]], u: float) -> str:
    total = sum(weight for _, weight in items)
    acc = 0.0
    for name, weight in items:
        acc += weight
        if u * total < acc:
            return name
    return items[-1][0]


def skill(request: LLMRequest) -> float:
    """Capability score of one request under the model described in the module docstring."""
    text = f"{request.system}\n{request.prompt}".lower()
    total = sum(weight for keyword, weight in FEATURE_WEIGHTS.items() if keyword in text)
    if request.model == "large":
        total += LARGE_MODEL_BONUS
    if "PLAN:" in request.prompt:
        total += PLAN_BONUS
    if "FEEDBACK:" in request.prompt:
        total += FEEDBACK_BONUS
    n_lines = sum(1 for line in request.system.splitlines() if line.startswith("- "))
    total -= INSTRUCTION_PENALTY * max(0, n_lines - INSTRUCTION_FREE_LINES)
    return total


def _mutation(
    op: str,
    role: str | None = None,
    value: str | int | None = None,
    index: int | None = None,
    rationale: str = "",
) -> dict[str, Any]:
    return {"op": op, "role": role, "value": value, "index": index, "rationale": rationale}


def _plan_mutations(
    genome: Mapping[str, Any], n: int, draw: Callable[..., float]
) -> list[dict[str, Any]]:
    """Propose ``n`` mutations that apply in order to ``genome``, each valid when reached.

    The plan works on a private copy of the genome, so later mutations see the effect of
    earlier ones. ``draw(*parts)`` supplies the randomness.
    """
    roles = [
        {
            "name": r["name"],
            "kind": r["kind"],
            "instructions": list(r.get("instructions", [])),
            "model": r.get("model", "small"),
        }
        for r in genome["roles"]
    ]
    state: dict[str, Any] = {"roles": roles, "max_retries": int(genome.get("max_retries", 1))}
    out: list[dict[str, Any]] = []
    for i in range(n):
        op = _weighted_pick(ARCH_OP_WEIGHTS, draw(i, "op"))
        mutation = None
        # An impossible op falls back to add_instruction, then set_retries (always possible).
        for candidate in (op, "add_instruction", "set_retries"):
            mutation = _apply_op(candidate, state, draw, (i, candidate))
            if mutation is not None:
                break
        assert mutation is not None
        out.append(mutation)
    return out


def _apply_op(
    op: str, state: dict[str, Any], draw: Callable[..., float], key: tuple[Any, ...]
) -> dict[str, Any] | None:
    """Apply ``op`` to ``state`` if it is possible; return the mutation or None."""
    roles: list[dict[str, Any]] = state["roles"]

    def pick(options: Sequence[Any], tag: str) -> Any:
        return _pick(options, draw(*key, tag))

    if op == "add_instruction":
        cands = [
            r
            for r in roles
            if len(r["instructions"]) < ARCH_MAX_INSTRUCTIONS
            and any(t not in r["instructions"] for t in INSTRUCTION_POOL)
        ]
        if not cands:
            return None
        role = pick(cands, "role")
        text = pick([t for t in INSTRUCTION_POOL if t not in role["instructions"]], "text")
        role["instructions"].append(text)
        return _mutation(
            op,
            role=role["name"],
            value=text,
            rationale=f"Try adding guidance to the {role['name']} role.",
        )

    if op == "replace_instruction":
        cands = [
            r
            for r in roles
            if r["instructions"] and any(t not in r["instructions"] for t in INSTRUCTION_POOL)
        ]
        if not cands:
            return None
        role = pick(cands, "role")
        index = pick(list(range(len(role["instructions"]))), "index")
        text = pick([t for t in INSTRUCTION_POOL if t not in role["instructions"]], "text")
        role["instructions"][index] = text
        return _mutation(
            op,
            role=role["name"],
            value=text,
            index=index,
            rationale=f"Swap an instruction on the {role['name']} role for other guidance.",
        )

    if op == "remove_instruction":
        cands = [r for r in roles if r["instructions"]]
        if not cands:
            return None
        role = pick(cands, "role")
        index = pick(list(range(len(role["instructions"]))), "index")
        del role["instructions"][index]
        return _mutation(
            op,
            role=role["name"],
            index=index,
            rationale=f"Drop an instruction from the {role['name']} role to save tokens.",
        )

    if op == "add_role":
        if len(roles) >= ARCH_MAX_ROLES:
            return None
        used = {r["name"] for r in roles}
        kinds = [
            k for k, limit in ARCH_KIND_LIMITS.items() if sum(r["kind"] == k for r in roles) < limit
        ]
        if not kinds:
            return None
        kind = pick(kinds, "kind")
        free = [name for name in (kind, kind + "2") if name not in used]
        if not free:
            return None
        roles.append({"name": free[0], "kind": kind, "instructions": [], "model": "small"})
        return _mutation(
            op,
            role=free[0],
            value=kind,
            rationale=f"Add a {kind} role to see whether it improves the outcome.",
        )

    if op == "remove_role":
        cands = [r for r in roles if r["kind"] != "coder"]
        if not cands:
            return None
        role = pick(cands, "role")
        roles.remove(role)
        return _mutation(
            op,
            role=role["name"],
            rationale=f"Remove the {role['name']} role to cut cost.",
        )

    if op == "change_model":
        role = pick(roles, "role")
        new = "large" if role["model"] == "small" else "small"
        role["model"] = new
        return _mutation(
            op,
            role=role["name"],
            value=new,
            rationale=f"Run the {role['name']} role on the {new} tier to test the trade-off.",
        )

    if op == "set_retries":
        options = [v for v in range(ARCH_MAX_RETRIES + 1) if v != state["max_retries"]]
        value = pick(options, "value")
        state["max_retries"] = value
        return _mutation(
            op,
            value=value,
            rationale=f"Change the feedback budget to {value} retries.",
        )

    raise ValueError(f"unknown mutation op {op!r}")


class SimulatedBackend:
    """Offline backend implementing the capability model in the module docstring.

    ``task_index`` maps task id to an object with ``reference_solution``, ``buggy_solutions``
    and ``difficulty`` attributes, such as ``bench.Task``.
    """

    name = "sim"

    def __init__(self, task_index: Mapping[str, Any], seed: int = 0) -> None:
        self.task_index = task_index
        self.seed = seed

    def _unit(self, *parts: object) -> float:
        """Deterministic float in [0, 1) derived from the seed and ``parts``."""
        key = f"{self.seed}|" + "|".join(map(str, parts))
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return float(int(digest[:12], 16)) / 16**12

    def complete(self, request: LLMRequest) -> LLMResponse:
        handlers = {
            "coder": self._coder,
            "planner": self._planner,
            "tester": self._tester,
            "reviewer": self._reviewer,
            "architect": self._architect,
        }
        handler = handlers.get(request.role_kind)
        if handler is None:
            raise ValueError(f"simulated backend has no behaviour for {request.role_kind!r}")
        text = handler(request)
        input_tokens = count_tokens_approx(request.system + request.prompt)
        output_tokens = count_tokens_approx(text)
        cost = estimate_cost(request.model, input_tokens, output_tokens)
        return LLMResponse(
            text=text, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost
        )

    def _coder(self, request: LLMRequest) -> str:
        task = self.task_index[request.task_id]  # KeyError for an unknown id
        p_success = _sigmoid(skill(request) + BASE_SKILL - task.difficulty)
        u = self._unit("coder", request.task_id, request.attempt, request.system, request.prompt)
        if u < p_success:
            code = task.reference_solution
        else:
            v = self._unit("coder", "bug", request.task_id, request.attempt, request.system,
                           request.prompt)
            code = _pick(list(task.buggy_solutions), v)
        return f"```python\n{code}\n```"

    def _planner(self, request: LLMRequest) -> str:
        system = request.system.lower()
        focus = [kw for kw, weight in FEATURE_WEIGHTS.items() if weight > 0 and kw in system]
        lines = [
            "PLAN:",
            "1. Restate the required behaviour.",
            "2. Write the function with the required signature.",
            "3. Check the result against the examples.",
        ]
        if focus:
            lines.append("Focus: " + ", ".join(focus))
        return "\n".join(lines)

    def _tester(self, request: LLMRequest) -> str:
        failing = [
            line.strip()
            for line in request.prompt.splitlines()
            if line.strip().startswith("assert")
        ]
        return "FEEDBACK: " + ("; ".join(failing) if failing else "tests failed")

    def _reviewer(self, request: LLMRequest) -> str:
        task = self.task_index[request.task_id]  # KeyError for an unknown id
        match = _FENCE.search(request.prompt)
        code = (match.group(1) if match else request.prompt).strip()
        buggy = {b.strip() for b in task.buggy_solutions}
        if code in buggy:
            catch_rate = _sigmoid(skill(request) - REVIEW_BIAS)
        else:
            catch_rate = REVIEW_FALSE_ALARM
        u = self._unit("reviewer", request.task_id, request.attempt, request.system,
                       request.prompt)
        return _REVIEW_HINT if u < catch_rate else "APPROVE"

    def _architect(self, request: LLMRequest) -> str:
        genome = request.metadata["genome"]
        n = int(request.metadata.get("n", 1))
        base = (json.dumps(genome, sort_keys=True), request.system, request.prompt, n)

        def draw(*parts: object) -> float:
            return self._unit("architect", *base, *parts)

        return json.dumps({"mutations": _plan_mutations(genome, n, draw)})
