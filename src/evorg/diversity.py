"""Structural distance between organization genomes, and the diversity of a population.

Two genomes are compared on what they structurally are: their role sets (name and kind), each
shared role's instruction set and model tier, and the retry budget. Ids, generation, parent ids
and mutation notes are ignored, and so is the order of roles and instructions, which the
metric treats as sets.

``genome_distance`` is the mean of four components, each in [0, 1]:

* roles: Jaccard distance between the sets of (name, kind) pairs;
* instructions: per role name in either genome, Jaccard distance between the two instruction
  sets, or 1 when the role exists in only one genome;
* models: per role name, 0 when the tier matches and 1 when it differs or the role is missing
  from one genome;
* retries: the absolute difference of the retry budgets, divided by ``MAX_RETRIES``.

The metric is symmetric and stays in [0, 1]. It is 0 exactly when two genomes have the same
``structure_key``, which is also what ``population_diversity`` counts as a distinct structure.
This module makes no triangle-inequality claim.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from itertools import combinations
from typing import NamedTuple

from evorg.genome import MAX_RETRIES, OrgGenome


class PopulationDiversity(NamedTuple):
    """Distinct structures in a group of genomes, and the mean distance over all pairs."""

    distinct: int
    mean_distance: float


def structure_key(genome: OrgGenome) -> Hashable:
    """A hashable key that is equal for two genomes exactly when their structure is equal.

    Roles are keyed by name, kind, model and instruction set, and the key holds the retry
    budget. Role and instruction order is ignored, so this is coarser than ``genome_id``.
    """
    roles = frozenset(
        (role.name, role.kind, role.model, frozenset(role.instructions)) for role in genome.roles
    )
    return roles, genome.max_retries


def _jaccard_distance(a: frozenset | set, b: frozenset | set) -> float:
    """1 minus the share of the union that both sets hold; 0 when both are empty."""
    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / len(union)


def genome_distance(a: OrgGenome, b: OrgGenome) -> float:
    """Structural distance between two genomes, in [0, 1]. Symmetric; 0 for the same structure."""
    a_roles = {role.name: role for role in a.roles}
    b_roles = {role.name: role for role in b.roles}
    names = set(a_roles) | set(b_roles)
    shared = set(a_roles) & set(b_roles)

    role_pairs_a = frozenset((role.name, role.kind) for role in a.roles)
    role_pairs_b = frozenset((role.name, role.kind) for role in b.roles)
    roles = _jaccard_distance(role_pairs_a, role_pairs_b)

    instruction_terms = [
        _jaccard_distance(set(a_roles[name].instructions), set(b_roles[name].instructions))
        if name in shared
        else 1.0
        for name in names
    ]
    model_terms = [
        float(a_roles[name].model != b_roles[name].model) if name in shared else 1.0
        for name in names
    ]
    retries = min(1.0, abs(a.max_retries - b.max_retries) / MAX_RETRIES)
    return (roles + _mean(instruction_terms) + _mean(model_terms) + retries) / 4


def _mean(values: Sequence[float]) -> float:
    """Mean of ``values``; 0 for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


def population_diversity(genomes: Sequence[OrgGenome]) -> PopulationDiversity:
    """Number of distinct structures and the mean ``genome_distance`` over all unordered pairs.

    Repeated genomes count once in ``distinct`` and add zero distance to the mean, since a pair
    of identical structures is at distance 0. With fewer than two genomes the mean is 0.
    """
    distinct = len({structure_key(genome) for genome in genomes})
    pairs = list(combinations(genomes, 2))
    if not pairs:
        return PopulationDiversity(distinct, 0.0)
    total = sum(genome_distance(a, b) for a, b in pairs)
    return PopulationDiversity(distinct, total / len(pairs))
