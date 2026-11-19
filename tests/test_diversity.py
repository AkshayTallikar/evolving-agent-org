import random
from dataclasses import replace
from itertools import product

import pytest

from evorg.diversity import genome_distance, population_diversity, structure_key
from evorg.genome import OrgGenome, RoleSpec, seed_genome
from evorg.mutation import apply_mutation, random_mutation


def _random_genomes(n: int, seed: int) -> list[OrgGenome]:
    """Valid genomes reached by random mutation chains from the seed, for property checks."""
    rng = random.Random(seed)
    genomes = [seed_genome()]
    while len(genomes) < n:
        base = rng.choice(genomes)
        genomes.append(apply_mutation(base, random_mutation(base, rng)))
    return genomes


def _with_lineage(genome: OrgGenome, **changes) -> OrgGenome:
    return replace(genome, **changes)


def _two_role_genome(**changes) -> OrgGenome:
    base = OrgGenome(
        roles=(
            RoleSpec("coder", "coder", ("Write a correct Python function.",), "small"),
            RoleSpec("reviewer", "reviewer", ("Look for bugs.", "Be concise."), "small"),
        ),
        max_retries=1,
    )
    return replace(base, **changes)


def test_identical_genome_is_at_distance_zero():
    genome = seed_genome()
    assert genome_distance(genome, genome) == 0.0


def test_lineage_and_notes_do_not_change_the_distance():
    genome = _two_role_genome()
    relabelled = _with_lineage(
        genome, parent_ids=("abc",), generation=7, mutation_note="anything at all"
    )
    assert genome_distance(genome, relabelled) == 0.0
    assert structure_key(genome) == structure_key(relabelled)


def test_role_and_instruction_order_do_not_change_the_distance():
    genome = _two_role_genome()
    reordered = replace(genome, roles=tuple(reversed(genome.roles)))
    assert genome_distance(genome, reordered) == 0.0


@pytest.mark.parametrize(
    "changed",
    [
        lambda g: replace(g, max_retries=3),
        lambda g: replace(g, roles=(replace(g.roles[0], model="large"), g.roles[1])),
        lambda g: replace(g, roles=(replace(g.roles[0], instructions=("Be brief.",)), g.roles[1])),
        lambda g: replace(g, roles=g.roles[:1]),
        lambda g: replace(g, roles=(*g.roles, RoleSpec("tester", "tester"))),
    ],
)
def test_each_structural_change_moves_the_distance_off_zero(changed):
    genome = _two_role_genome()
    other = changed(genome)
    other.validate()
    assert genome_distance(genome, other) > 0.0
    assert structure_key(genome) != structure_key(other)


def test_distance_is_symmetric_and_stays_in_the_unit_interval():
    genomes = _random_genomes(30, seed=4)
    for a, b in product(genomes, repeat=2):
        forward = genome_distance(a, b)
        assert forward == genome_distance(b, a)
        assert 0.0 <= forward <= 1.0


def test_distance_is_zero_exactly_when_the_structure_key_matches():
    genomes = _random_genomes(25, seed=9)
    for a, b in product(genomes, repeat=2):
        same_structure = structure_key(a) == structure_key(b)
        assert (genome_distance(a, b) == 0.0) == same_structure


def test_a_role_missing_from_one_side_is_maximally_different_for_that_role():
    genome = _two_role_genome()
    only_coder = replace(genome, roles=genome.roles[:1])
    # Roles: {coder, reviewer} vs {coder} -> Jaccard 1/2. Instructions: coder shared (0),
    # reviewer missing (1) -> mean 1/2. Models: coder shared (0), reviewer missing (1) -> 1/2.
    # Retries equal -> 0. Mean of the four: (0.5 + 0.5 + 0.5 + 0) / 4.
    assert genome_distance(genome, only_coder) == pytest.approx(0.375)


def test_population_diversity_counts_structures_and_averages_pairs():
    base = seed_genome()
    other = replace(base, max_retries=3)
    genomes = [base, base, other]
    stats = population_diversity(genomes)
    assert stats.distinct == 2
    pair_mean = (
        genome_distance(base, base) + genome_distance(base, other) + genome_distance(base, other)
    ) / 3
    assert stats.mean_distance == pytest.approx(pair_mean)


def test_population_diversity_of_identical_or_tiny_groups():
    assert population_diversity([]) == (0, 0.0)
    assert population_diversity([seed_genome()]) == (1, 0.0)
    assert population_diversity([seed_genome(), seed_genome()]).mean_distance == 0.0
