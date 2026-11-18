"""Organization genomes: roles, instructions, model tiers and retry budget.

A genome is the unit that evolution mutates and selects. Its identity (``genome_id``)
covers what the organization *does* (roles and retries). Lineage fields such as
``parent_ids`` and ``generation`` describe *how* it came to exist and do not change it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from evorg.genes import MODEL_TIERS, ROLE_KINDS

MAX_ROLES = 6
MAX_INSTRUCTIONS = 8
MAX_INSTRUCTION_CHARS = 200
MAX_RETRIES = 4
MAX_PER_KIND = {"planner": 2, "coder": 1, "tester": 1, "reviewer": 1}

_PERSONAS: dict[str, str] = {
    "planner": (
        "You are the planner on a small software team. You break a coding task into a short "
        "numbered plan."
    ),
    "coder": (
        "You are the coder on a small software team. You write a single correct Python function."
    ),
    "tester": (
        "You are the tester on a small software team. You turn failing test output into "
        "concrete feedback for the coder."
    ),
    "reviewer": (
        "You are the reviewer on a small software team. You read a candidate function and "
        "look for bugs."
    ),
}


class GenomeError(ValueError):
    """Raised when a genome breaks one of the validation rules."""


@dataclass(frozen=True)
class RoleSpec:
    """One role in an organization: its name, kind, instructions and model tier."""

    name: str
    kind: str
    instructions: tuple[str, ...] = ()
    model: str = "small"

    def system_prompt(self) -> str:
        """Return the fixed persona for this kind, followed by the instructions as bullets."""
        lines = [_PERSONAS[self.kind]]
        if self.instructions:
            lines.append("Instructions:")
            lines.extend(f"- {text}" for text in self.instructions)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "instructions": list(self.instructions),
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RoleSpec:
        return cls(
            name=d["name"],
            kind=d["kind"],
            instructions=tuple(d.get("instructions", ())),
            model=d.get("model", "small"),
        )


@dataclass(frozen=True)
class OrgGenome:
    """A whole organization: its roles plus the feedback-loop budget and lineage."""

    roles: tuple[RoleSpec, ...]
    max_retries: int = 1
    parent_ids: tuple[str, ...] = ()
    generation: int = 0
    mutation_note: str = ""

    @property
    def genome_id(self) -> str:
        """First 10 hex chars of a sha1 over the canonical roles and retry count only."""
        payload = {
            "roles": [role.to_dict() for role in self.roles],
            "max_retries": self.max_retries,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:10]

    def role(self, name: str) -> RoleSpec:
        """Return the role with this name. Raises KeyError if there is none."""
        for role in self.roles:
            if role.name == name:
                return role
        raise KeyError(name)

    def roles_of_kind(self, kind: str) -> tuple[RoleSpec, ...]:
        return tuple(role for role in self.roles if role.kind == kind)

    def validate(self) -> None:
        """Check every rule in design section 4.2. Raises GenomeError on the first failure."""
        if not 1 <= len(self.roles) <= MAX_ROLES:
            raise GenomeError(
                f"an organization needs 1 to {MAX_ROLES} roles, got {len(self.roles)}"
            )

        seen: set[str] = set()
        for role in self.roles:
            if not role.name.strip():
                raise GenomeError("role names must be non-empty")
            if role.name in seen:
                raise GenomeError(f"duplicate role name '{role.name}'")
            seen.add(role.name)
            if role.kind not in ROLE_KINDS:
                raise GenomeError(
                    f"role '{role.name}' has unknown kind '{role.kind}'; "
                    f"expected one of {', '.join(ROLE_KINDS)}"
                )
            if role.model not in MODEL_TIERS:
                raise GenomeError(
                    f"role '{role.name}' has unknown model '{role.model}'; "
                    f"expected one of {', '.join(MODEL_TIERS)}"
                )
            _validate_instructions(role)

        for kind, limit in MAX_PER_KIND.items():
            count = len(self.roles_of_kind(kind))
            if kind == "coder":
                if count != 1:
                    raise GenomeError(f"an organization needs exactly one coder, got {count}")
            elif count > limit:
                raise GenomeError(f"at most {limit} {kind} allowed, got {count}")

        if not 0 <= self.max_retries <= MAX_RETRIES:
            raise GenomeError(
                f"max_retries must be between 0 and {MAX_RETRIES}, got {self.max_retries}"
            )

    def to_dict(self) -> dict:
        return {
            "roles": [role.to_dict() for role in self.roles],
            "max_retries": self.max_retries,
            "parent_ids": list(self.parent_ids),
            "generation": self.generation,
            "mutation_note": self.mutation_note,
            "genome_id": self.genome_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> OrgGenome:
        """Rebuild a genome from ``to_dict`` output. The stored ``genome_id`` is ignored."""
        return cls(
            roles=tuple(RoleSpec.from_dict(r) for r in d["roles"]),
            max_retries=d.get("max_retries", 1),
            parent_ids=tuple(d.get("parent_ids", ())),
            generation=d.get("generation", 0),
            mutation_note=d.get("mutation_note", ""),
        )


def _validate_instructions(role: RoleSpec) -> None:
    if len(role.instructions) > MAX_INSTRUCTIONS:
        raise GenomeError(
            f"role '{role.name}' has {len(role.instructions)} instructions; "
            f"at most {MAX_INSTRUCTIONS} allowed"
        )
    seen: set[str] = set()
    for text in role.instructions:
        if not 1 <= len(text) <= MAX_INSTRUCTION_CHARS or not text.strip():
            raise GenomeError(
                f"role '{role.name}' has an instruction of {len(text)} characters; "
                f"each must be 1 to {MAX_INSTRUCTION_CHARS} characters and not blank"
            )
        if text in seen:
            raise GenomeError(f"role '{role.name}' repeats the instruction '{text}'")
        seen.add(text)


def diff_genomes(a: OrgGenome, b: OrgGenome) -> list[str]:
    """Readable lines describing how genome ``b`` differs from ``a``. Empty if equal.

    Roles are matched by name. Output order: added roles, removed roles, per-role changes
    (in ``b``'s role order), then the retry budget.
    """
    lines: list[str] = []
    a_names = {role.name for role in a.roles}
    b_names = {role.name for role in b.roles}

    for role in b.roles:
        if role.name not in a_names:
            lines.append(f"+ role {role.name} ({role.kind}, {role.model})")
    for role in a.roles:
        if role.name not in b_names:
            lines.append(f"- role {role.name} ({role.kind}, {role.model})")

    for new in b.roles:
        if new.name not in a_names:
            continue
        old = a.role(new.name)
        if old.kind != new.kind:
            lines.append(f"{new.name}: kind {old.kind} -> {new.kind}")
        if old.model != new.model:
            lines.append(f"{new.name}: model {old.model} -> {new.model}")
        for text in new.instructions:
            if text not in old.instructions:
                lines.append(f"{new.name}: + instruction '{text}'")
        for text in old.instructions:
            if text not in new.instructions:
                lines.append(f"{new.name}: - instruction '{text}'")

    if a.max_retries != b.max_retries:
        lines.append(f"max_retries {a.max_retries} -> {b.max_retries}")
    return lines


def seed_genome() -> OrgGenome:
    """The starting organization: a single small-model coder with one instruction."""
    return OrgGenome(
        roles=(
            RoleSpec(
                name="coder",
                kind="coder",
                instructions=("Write a correct Python function.",),
                model="small",
            ),
        ),
        max_retries=1,
    )
