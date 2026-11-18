"""Tests for evorg.crossover: role-aligned uniform crossover of two genomes."""

from __future__ import annotations

import random

import pytest

import evorg.crossover as crossover_module
from evorg.crossover import crossover
from evorg.genes import INSTRUCTION_POOL
from evorg.genome import MAX_INSTRUCTIONS, GenomeError, OrgGenome, RoleSpec, seed_genome
from evorg.mutation import apply_mutation, random_mutation


def role(name: str, kind: str, *instructions: str, model: str = "small") -> RoleSpec:
    return RoleSpec(name=name, kind=kind, instructions=tuple(instructions), model=model)


def random_genome(rng: random.Random, steps: int = 6) -> OrgGenome:
    """A valid genome grown from the seed by a few random mutations."""
    genome = seed_genome()
    for _ in range(steps):
        genome = apply_mutation(genome, random_mutation(genome, rng))
    return genome


def parent_pairs(count: int, seed: int) -> list[tuple[OrgGenome, OrgGenome]]:
    rng = random.Random(seed)
    return [(random_genome(rng), random_genome(rng)) for _ in range(count)]


def test_same_seed_gives_the_same_child():
    a, b = parent_pairs(1, seed=1)[0]
    first = [crossover(a, b, random.Random(s)) for s in range(30)]
    second = [crossover(a, b, random.Random(s)) for s in range(30)]
    assert first == second


def test_child_always_validates_over_random_pairs():
    rng = random.Random(2)
    for _ in range(500):
        a, b = random_genome(rng), random_genome(rng)
        crossover(a, b, rng).validate()


def test_crossing_a_genome_with_itself_keeps_roles_and_instructions():
    rng = random.Random(3)
    for _ in range(50):
        genome = random_genome(rng)
        child = crossover(genome, genome, rng)
        assert child.roles == genome.roles
        assert child.max_retries == genome.max_retries
        assert child.genome_id == genome.genome_id


def test_every_child_role_comes_from_a_parent():
    rng = random.Random(4)
    for _ in range(200):
        a, b = random_genome(rng), random_genome(rng)
        child = crossover(a, b, rng)
        union = a.roles + b.roles
        coder_union = {text for r in union if r.kind == "coder" for text in r.instructions}
        for child_role in child.roles:
            if child_role.kind == "coder":
                assert set(child_role.instructions) <= coder_union
                continue
            assert any(
                p.kind == child_role.kind
                and p.model == child_role.model
                and p.instructions == child_role.instructions
                for p in union
            )


def test_child_lineage_fields():
    a = seed_genome()
    b = apply_mutation(a, random_mutation(a, random.Random(5)))
    b = apply_mutation(b, random_mutation(b, random.Random(6)))
    child = crossover(a, b, random.Random(7))
    assert child.parent_ids == (a.genome_id, b.genome_id)
    assert child.generation == max(a.generation, b.generation) + 1
    assert child.mutation_note == f"crossover of {a.genome_id} x {b.genome_id}"


def test_colliding_role_names_are_renamed_uniquely():
    # Parent a has a tester named "tester"; parent b has a planner named "tester". When both
    # are taken, the planner keeps the name and the tester is renamed.
    a = OrgGenome(roles=(role("coder", "coder", "Write code."), role("tester", "tester")))
    b = OrgGenome(roles=(role("tester", "planner"), role("coder", "coder", "Write code.")))
    renamed = []
    for seed in range(300):
        child = crossover(a, b, random.Random(seed))
        names = [r.name for r in child.roles]
        assert len(names) == len(set(names))
        renamed += [r for r in child.roles if r.kind == "tester" and r.name != "tester"]
    assert renamed
    assert all(r.name == "tester2" for r in renamed)


def test_instruction_cap_is_respected_and_reached():
    pool = list(INSTRUCTION_POOL)
    a = OrgGenome(roles=(role("coder", "coder", *pool[:6]),))
    b = OrgGenome(roles=(role("coder", "coder", *pool[6:12]),))
    lengths = set()
    for seed in range(200):
        coder = crossover(a, b, random.Random(seed)).roles[0]
        assert len(coder.instructions) <= MAX_INSTRUCTIONS
        lengths.add(len(coder.instructions))
    assert MAX_INSTRUCTIONS in lengths


def test_coder_instructions_mix_across_parents():
    a = OrgGenome(roles=(role("coder", "coder", "Write code.", "Use type hints."),))
    b = OrgGenome(roles=(role("coder", "coder", "Handle empty input explicitly."),))
    mixed = 0
    for seed in range(100):
        child = crossover(a, b, random.Random(seed))
        instructions = child.roles[0].instructions
        assert len(instructions) == len(set(instructions))
        assert set(instructions) <= set(a.roles[0].instructions) | set(b.roles[0].instructions)
        if instructions not in (a.roles[0].instructions, b.roles[0].instructions):
            mixed += 1
    assert mixed > 0


def test_children_often_differ_from_both_parents():
    pairs = parent_pairs(20, seed=8)
    distinct = 0
    for i, (a, b) in enumerate(pairs):
        for seed in range(10):
            child = crossover(a, b, random.Random(seed * 100 + i))
            if child.genome_id not in (a.genome_id, b.genome_id):
                distinct += 1
    assert distinct > 0


def test_invalid_child_falls_back_to_a_copy_of_a(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(crossover_module, "_is_valid", lambda child: False)
    a, b = parent_pairs(1, seed=9)[0]
    child = crossover(a, b, random.Random(0))
    assert child.roles == a.roles
    assert child.max_retries == a.max_retries
    assert child.parent_ids == (a.genome_id, b.genome_id)
    assert child.generation == max(a.generation, b.generation) + 1
    assert child.mutation_note == f"crossover of {a.genome_id} x {b.genome_id}"


def test_invalid_parent_is_rejected():
    bad = OrgGenome(roles=(role("tester", "tester"),))
    with pytest.raises(GenomeError, match="exactly one coder"):
        crossover(bad, seed_genome(), random.Random(0))
