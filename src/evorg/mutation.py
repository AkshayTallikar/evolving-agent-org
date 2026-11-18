"""Mutations: named, single edits to an organization genome.

A mutation is applied to a parent genome to produce an independent child. The child records
its lineage (parent id, generation, and a one-line note built from the mutation), so every
champion can be traced back to the seed one step at a time. The architect that proposes
mutations lives further down this module.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

from evorg.genes import INSTRUCTION_POOL, MODEL_TIERS, ROLE_KINDS
from evorg.genome import (
    MAX_INSTRUCTION_CHARS,
    MAX_INSTRUCTIONS,
    MAX_RETRIES,
    MAX_ROLES,
    GenomeError,
    OrgGenome,
    RoleSpec,
)
from evorg.llm.base import LLMBackend, LLMRequest

OPS: tuple[str, ...] = (
    "add_instruction",
    "remove_instruction",
    "replace_instruction",
    "add_role",
    "remove_role",
    "change_model",
    "set_retries",
)

RANDOM_TRIES = 50
RANDOM_RATIONALE = "random exploration"


class MutationError(ValueError):
    """Raised when a mutation cannot be applied to a genome."""


@dataclass(frozen=True)
class Mutation:
    """One edit. Which fields matter depends on ``op``; see ``apply_mutation``."""

    op: str
    role: str | None = None
    value: str | int | None = None
    index: int | None = None
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "role": self.role,
            "value": self.value,
            "index": self.index,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Mutation:
        """Rebuild a mutation from a dict, tolerating missing keys and loose integers.

        This does not check the mutation against any genome; use ``apply_mutation`` for that.
        Raises MutationError only when ``d`` is not a dict at all.
        """
        if not isinstance(d, dict):
            raise MutationError(f"a mutation must be an object, got {type(d).__name__}")
        op = d.get("op")
        role = d.get("role")
        value = d.get("value")
        index = _coerce_int(d.get("index"))
        rationale = d.get("rationale")
        if op == "set_retries" and _coerce_int(value) is not None:
            value = _coerce_int(value)
        return cls(
            op="" if op is None else str(op),
            role=None if role is None else str(role),
            value=value,
            index=index,
            rationale="" if rationale is None else str(rationale),
        )

    def describe(self) -> str:
        """One short line for lineage reports. Does not include the rationale."""
        role = self.role or "?"
        if self.op == "add_instruction":
            return f"{role}: + instruction '{self.value}'"
        if self.op == "remove_instruction":
            return f"{role}: - instruction {self.index}"
        if self.op == "replace_instruction":
            return f"{role}: replace instruction {self.index} with '{self.value}'"
        if self.op == "add_role":
            return f"+ role {role} ({self.value})"
        if self.op == "remove_role":
            return f"- role {role}"
        if self.op == "change_model":
            return f"{role}: model -> {self.value}"
        if self.op == "set_retries":
            return f"max_retries -> {self.value}"
        return f"unknown op {self.op!r}"


def resolve(genome: OrgGenome, m: Mutation) -> Mutation:
    """Return ``m`` with its open fields filled in from ``genome``.

    An ``add_role`` without a name gets its kind as the name, or kind + "2" when the kind is
    taken. A ``change_model`` without a value flips the role's tier. Anything else, or
    anything that cannot be resolved, comes back unchanged; ``apply_mutation`` then rejects it.
    """
    if m.op == "add_role" and m.role is None and m.value in ROLE_KINDS:
        taken = {role.name for role in genome.roles}
        for name in (str(m.value), f"{m.value}2"):
            if name not in taken:
                return replace(m, role=name)
        return m
    if m.op == "change_model" and m.value is None:
        for role in genome.roles:
            if role.name == m.role:
                flipped = "large" if role.model == "small" else "small"
                return replace(m, value=flipped)
    return m


def apply_mutation(genome: OrgGenome, m: Mutation) -> OrgGenome:
    """Return a validated child of ``genome`` with ``m`` applied.

    Raises MutationError when the mutation does not apply: unknown op, missing role, bad
    index, a rule of design section 4.2 would break, or a change that leaves the genome id
    unchanged (a no-op).
    """
    if m.op not in OPS:
        raise MutationError(f"unknown op {m.op!r}; expected one of {', '.join(OPS)}")
    m = resolve(genome, m)
    roles = list(genome.roles)
    retries = genome.max_retries

    if m.op == "add_instruction":
        i = _role_index(genome, m.role)
        text = _require_text(m.value)
        role = roles[i]
        roles[i] = replace(role, instructions=(*role.instructions, text))

    elif m.op == "remove_instruction":
        i = _role_index(genome, m.role)
        role = roles[i]
        j = _require_index(m.index, role)
        roles[i] = replace(role, instructions=role.instructions[:j] + role.instructions[j + 1 :])

    elif m.op == "replace_instruction":
        i = _role_index(genome, m.role)
        role = roles[i]
        j = _require_index(m.index, role)
        text = _require_text(m.value)
        roles[i] = replace(
            role, instructions=role.instructions[:j] + (text,) + role.instructions[j + 1 :]
        )

    elif m.op == "add_role":
        if m.value not in ROLE_KINDS:
            raise MutationError(f"new role kind {m.value!r} is not one of {', '.join(ROLE_KINDS)}")
        if m.value == "coder":
            raise MutationError("an organization has exactly one coder; it cannot add another")
        if not isinstance(m.role, str) or not m.role.strip():
            raise MutationError("a new role needs a non-empty name")
        if any(role.name == m.role for role in roles):
            raise MutationError(f"a role named {m.role!r} already exists")
        new = RoleSpec(name=m.role, kind=str(m.value), instructions=(), model="small")
        if new.kind == "planner":
            # Planners run before the coder, in role order, and their plans feed the coder.
            position = next((k for k, r in enumerate(roles) if r.kind == "coder"), len(roles))
            roles.insert(position, new)
        else:
            roles.append(new)

    elif m.op == "remove_role":
        i = _role_index(genome, m.role)
        if roles[i].kind == "coder":
            raise MutationError("the coder cannot be removed")
        del roles[i]

    elif m.op == "change_model":
        i = _role_index(genome, m.role)
        role = roles[i]
        target = m.value
        if target is None:
            target = "large" if role.model == "small" else "small"
        if target not in MODEL_TIERS:
            raise MutationError(f"model {target!r} is not one of {', '.join(MODEL_TIERS)}")
        if target == role.model:
            raise MutationError(f"role '{role.name}' already runs on {target}")
        roles[i] = replace(role, model=target)

    elif m.op == "set_retries":
        if not _is_int(m.value):
            raise MutationError(f"set_retries needs an integer value, got {m.value!r}")
        retries = int(m.value)

    child = replace(
        genome,
        roles=tuple(roles),
        max_retries=retries,
        parent_ids=(genome.genome_id,),
        generation=genome.generation + 1,
        mutation_note=_note(m),
    )
    try:
        child.validate()
    except GenomeError as exc:
        raise MutationError(str(exc)) from exc
    if child.genome_id == genome.genome_id:
        raise MutationError(f"mutation changes nothing: {m.describe()}")
    return child


def random_mutation(genome: OrgGenome, rng: random.Random) -> Mutation:
    """Draw a mutation that applies cleanly to ``genome``.

    Draws ops and gene-pool instructions from ``rng`` and keeps the first candidate that
    applies, trying at most ``RANDOM_TRIES`` times. The fallback changes the retry budget to a
    different value, which is always valid.
    """
    for _ in range(RANDOM_TRIES):
        candidate = _random_candidate(genome, rng.choice(OPS), rng)
        if candidate is None:
            continue
        try:
            apply_mutation(genome, candidate)
        except MutationError:
            continue
        return resolve(genome, candidate)

    current = genome.max_retries
    options = [v for v in range(MAX_RETRIES + 1) if v != current]
    fallback = Mutation("set_retries", value=rng.choice(options), rationale=RANDOM_RATIONALE)
    apply_mutation(genome, fallback)
    return fallback


def _random_candidate(genome: OrgGenome, op: str, rng: random.Random) -> Mutation | None:
    """One unchecked random mutation of kind ``op``, or None if the op has no target."""
    role = rng.choice(genome.roles)
    if op == "add_instruction":
        return Mutation(
            op, role=role.name, value=rng.choice(INSTRUCTION_POOL), rationale=RANDOM_RATIONALE
        )
    if op == "replace_instruction":
        if not role.instructions:
            return None
        return Mutation(
            op,
            role=role.name,
            value=rng.choice(INSTRUCTION_POOL),
            index=rng.randrange(len(role.instructions)),
            rationale=RANDOM_RATIONALE,
        )
    if op == "remove_instruction":
        if not role.instructions:
            return None
        return Mutation(
            op,
            role=role.name,
            index=rng.randrange(len(role.instructions)),
            rationale=RANDOM_RATIONALE,
        )
    if op == "add_role":
        kind = rng.choice([k for k in ROLE_KINDS if k != "coder"])
        return Mutation(op, value=kind, rationale=RANDOM_RATIONALE)
    if op == "remove_role":
        removable = [r for r in genome.roles if r.kind != "coder"]
        if not removable:
            return None
        return Mutation(op, role=rng.choice(removable).name, rationale=RANDOM_RATIONALE)
    if op == "change_model":
        return Mutation(op, role=role.name, rationale=RANDOM_RATIONALE)
    if op == "set_retries":
        return Mutation(op, value=rng.randrange(MAX_RETRIES + 1), rationale=RANDOM_RATIONALE)
    return None


def _role_index(genome: OrgGenome, name: str | None) -> int:
    if name is None:
        raise MutationError("this op needs a role name")
    for i, role in enumerate(genome.roles):
        if role.name == name:
            return i
    raise MutationError(f"no role named {name!r}")


def _require_index(index: int | None, role: RoleSpec) -> int:
    if not _is_int(index) or not 0 <= index < len(role.instructions):
        raise MutationError(
            f"role '{role.name}' has no instruction at index {index!r} "
            f"({len(role.instructions)} instructions)"
        )
    return int(index)


def _require_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MutationError("instruction text must be a non-empty string")
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _coerce_int(value: Any) -> int | None:
    """An int for ints, integral floats and numeric strings; None for anything else."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


# --- architect ----------------------------------------------------------------------------

_SYSTEM = (
    "You are the architect of an AI software team. You propose changes to the team's roles, "
    "instructions, model tiers and retry budget that will raise the pass rate on a coding "
    "benchmark without wasting cost. Reply with JSON only, no prose."
)

_SCHEMA = """\
Each mutation is an object with the keys "op", "role", "value", "index" and "rationale".
Use null for a key that an op does not need. The ops are:
- add_instruction: role = role name; value = instruction text, appended to that role.
- remove_instruction: role = role name; index = 0-based position of the instruction to drop.
- replace_instruction: role = role name; index = 0-based position; value = new instruction text.
- add_role: value = kind (planner, tester or reviewer); role = name for the new role.
  The new role starts with no instructions on the small tier. Planners run before the coder.
- remove_role: role = name of a role to remove (never the coder).
- change_model: role = role name; value = small or large.
- set_retries: value = integer retry budget; role and index are null.
"rationale" is one sentence saying why the change should help."""


def _limits() -> list[str]:
    return [
        f"between 1 and {MAX_ROLES} roles; role names unique and non-empty",
        "kinds: planner, coder, tester, reviewer",
        "exactly one coder; at most 2 planners, 1 tester and 1 reviewer",
        "models: small or large",
        f"at most {MAX_INSTRUCTIONS} instructions per role, each 1 to {MAX_INSTRUCTION_CHARS} "
        "characters, no duplicates within a role",
        f"max_retries between 0 and {MAX_RETRIES}",
    ]


def _build_prompt(genome: OrgGenome, report: str, n: int) -> str:
    limits = "\n".join(f"- {line}" for line in _limits())
    return "\n\n".join(
        [
            "Current organization (genome JSON):",
            json.dumps(genome.to_dict(), indent=2),
            "Evaluation report:",
            report.strip() or "(no report)",
            "Mutation schema:",
            _SCHEMA,
            "Validation limits the organization must stay within:",
            limits,
            f'Propose exactly {n} mutations as {{"mutations": [...]}}. Each one is an '
            "independent alternative applied to the current organization above, not a step "
            "in a chain. Reply with JSON only.",
        ]
    )


_FENCE = re.compile(r"```[ \t]*(?:json)?[ \t]*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def _candidate_texts(text: str) -> Iterator[str]:
    """Strings that might hold the JSON, most likely first."""
    yield text.strip()
    for match in _FENCE.finditer(text):
        yield match.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            yield text[start : end + 1]


def _parse_items(text: str) -> list[Any]:
    """The mutation dicts in an LLM reply, or [] when no JSON with mutations is found."""
    for candidate in _candidate_texts(text):
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict) and isinstance(data.get("mutations"), list):
            return data["mutations"]
        if isinstance(data, list):
            return data
    return []


class Architect:
    """Asks a large-model backend for mutations and keeps only those that apply.

    Each proposal is treated as an independent alternative applied to the same parent. Any
    shortfall is topped up with ``random_mutation``, so ``propose`` always returns exactly ``n``.
    ``stats`` accumulates across calls: ``calls``, ``proposed`` (items found in replies),
    ``valid`` (items that applied to the parent), ``fallback`` (random top-ups) and
    ``cost_usd``. Malformed output never raises; errors from the backend itself propagate.
    """

    def __init__(self, backend: LLMBackend, rng: random.Random, model: str = "large") -> None:
        self.backend = backend
        self.rng = rng
        self.model = model
        self.stats: dict[str, Any] = {
            "calls": 0,
            "proposed": 0,
            "valid": 0,
            "fallback": 0,
            "cost_usd": 0.0,
        }

    def propose(self, genome: OrgGenome, report: str, n: int = 1) -> list[Mutation]:
        """Return exactly ``n`` mutations that each apply to ``genome``."""
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        request = LLMRequest(
            role_kind="architect",
            model=self.model,
            system=_SYSTEM,
            prompt=_build_prompt(genome, report, n),
            metadata={"genome": genome.to_dict(), "n": n},
        )
        response = self.backend.complete(request)
        self.stats["calls"] += 1
        self.stats["cost_usd"] += float(response.cost_usd)

        items = _parse_items(response.text)
        self.stats["proposed"] += len(items)

        chosen: list[Mutation] = []
        seen: set[str] = set()
        for item in items:
            try:
                m = resolve(genome, Mutation.from_dict(item))
                child_id = apply_mutation(genome, m).genome_id
            except (MutationError, TypeError, ValueError):
                continue
            self.stats["valid"] += 1
            if len(chosen) < n and child_id not in seen:
                chosen.append(m)
                seen.add(child_id)

        while len(chosen) < n:
            m = self._top_up(genome, seen)
            chosen.append(m)
            seen.add(apply_mutation(genome, m).genome_id)
            self.stats["fallback"] += 1
        return chosen

    def _top_up(self, genome: OrgGenome, seen: set[str]) -> Mutation:
        """A random mutation whose child is not already in ``seen``, when one can be found."""
        m = random_mutation(genome, self.rng)
        for _ in range(RANDOM_TRIES):
            if apply_mutation(genome, m).genome_id not in seen:
                break
            m = random_mutation(genome, self.rng)
        return m


def _note(m: Mutation) -> str:
    """Lineage note: what changed, plus the proposer's reason when it gave one."""
    reason = m.rationale.strip()
    return f"{m.describe()}. Why: {reason}" if reason else m.describe()
