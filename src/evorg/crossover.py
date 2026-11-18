"""Role-aligned uniform crossover: combine two organization genomes into one child.

Roles are matched by their kind slot, not by name: the coder, the first and second planner
(in each parent's role order), the tester and the reviewer. Each slot is inherited whole from
one parent, chosen by a coin flip. The coder slot also mixes instruction lists. Parent ids
and generation are recorded so the child can be traced like a mutated genome.
"""

from __future__ import annotations

import random
from dataclasses import replace

from evorg.genome import MAX_INSTRUCTIONS, GenomeError, OrgGenome, RoleSpec

# Fixed order in which slots are drawn, so results depend only on the rng state.
_SLOT_ORDER: tuple[str, ...] = ("coder", "planner1", "planner2", "tester", "reviewer")

_SLOT_KINDS: tuple[str, ...] = ("coder", "tester", "reviewer")


def crossover(a: OrgGenome, b: OrgGenome, rng: random.Random) -> OrgGenome:
    """Return a child genome that mixes roles from parents ``a`` and ``b``.

    For each slot present in at least one parent, a fair coin picks the parent whose role
    is used. If that parent lacks the slot, the child lacks it too. The coder's instructions
    are mixed: the chosen parent's list comes first, then each instruction of the other coder
    that is not yet present is kept with probability 0.5, up to ``MAX_INSTRUCTIONS``. The
    retry budget is taken from one parent at random.

    Child roles are ordered planners first, then the rest in the order they had in their
    parent, so crossing a genome with itself gives back the same roles. Duplicate names are
    renamed with the lowest free numeric suffix (``tester2``, ``tester3``, ...).

    Lineage: ``parent_ids`` holds both parent ids, ``generation`` is one more than the
    newer parent, and ``mutation_note`` reads ``"crossover of <a> x <b>"``. The child is
    validated. If validation fails, which the rules above should prevent, the result is a copy
    of ``a`` with the same lineage fields. Both parents must already be valid genomes.

    The same ``rng`` state always gives the same child. Raises GenomeError if a parent is
    invalid.
    """
    a.validate()
    b.validate()
    a_slots = _slot_map(a)
    b_slots = _slot_map(b)

    picks: list[tuple[int, RoleSpec]] = []
    for slot in _SLOT_ORDER:
        if slot not in a_slots and slot not in b_slots:
            continue
        from_a = rng.random() < 0.5
        chosen, other = (a_slots, b_slots) if from_a else (b_slots, a_slots)
        if slot not in chosen:
            continue
        index, role = chosen[slot]
        if slot == "coder":
            role = _mix_coder(role, other["coder"][1], rng)
        picks.append((index, role))

    retries = a.max_retries if rng.random() < 0.5 else b.max_retries
    roles = _unique_names(_ordered(picks))
    lineage = {
        "parent_ids": (a.genome_id, b.genome_id),
        "generation": max(a.generation, b.generation) + 1,
        "mutation_note": f"crossover of {a.genome_id} x {b.genome_id}",
    }
    child = OrgGenome(roles=roles, max_retries=retries, **lineage)
    if _is_valid(child):
        return child
    return replace(a, **lineage)


def _slot_map(genome: OrgGenome) -> dict[str, tuple[int, RoleSpec]]:
    """Map each slot name to (position in the genome's role order, role).

    A valid genome has at most two planners, so ``planner1`` and ``planner2`` cover them all.
    """
    slots: dict[str, tuple[int, RoleSpec]] = {}
    planners = [(i, r) for i, r in enumerate(genome.roles) if r.kind == "planner"]
    for number, (index, role) in enumerate(planners[:2], start=1):
        slots[f"planner{number}"] = (index, role)
    for index, role in enumerate(genome.roles):
        if role.kind in _SLOT_KINDS:
            slots[role.kind] = (index, role)
    return slots


def _mix_coder(coder: RoleSpec, other: RoleSpec, rng: random.Random) -> RoleSpec:
    """The chosen coder with instructions from the other coder appended at random."""
    instructions = list(coder.instructions)
    for text in other.instructions:
        if len(instructions) >= MAX_INSTRUCTIONS:
            break
        if text in instructions:
            continue
        if rng.random() < 0.5:
            instructions.append(text)
    return replace(coder, instructions=tuple(instructions))


def _ordered(picks: list[tuple[int, RoleSpec]]) -> list[RoleSpec]:
    """Planners first, then the other roles, each group in its parent's role order."""
    ranked = sorted(picks, key=lambda pick: (pick[1].kind != "planner", pick[0]))
    return [role for _, role in ranked]


def _unique_names(roles: list[RoleSpec]) -> tuple[RoleSpec, ...]:
    """Keep the first role of each name. Rename later clashes to the lowest free suffix.

    Suffixes never reuse a name that any input role already has, so renaming cannot clash
    with a role that keeps its own name.
    """
    reserved = {role.name for role in roles}
    assigned: set[str] = set()
    named: list[RoleSpec] = []
    for role in roles:
        name = role.name
        if name in assigned:
            name = _free_name(role.name, reserved | assigned)
        assigned.add(name)
        named.append(replace(role, name=name))
    return tuple(named)


def _free_name(base: str, taken: set[str]) -> str:
    """``base`` with the lowest numeric suffix from 2 that is not in ``taken``."""
    number = 2
    while f"{base}{number}" in taken:
        number += 1
    return f"{base}{number}"


def _is_valid(child: OrgGenome) -> bool:
    try:
        child.validate()
    except GenomeError:
        return False
    return True
