import json
from dataclasses import replace

import pytest

from evorg import evolution
from evorg.cli import main
from evorg.diversity import population_diversity
from evorg.evolution import Evaluation, EvolutionConfig, _crowding_survivors, run_simulated
from evorg.genome import OrgGenome, RoleSpec, seed_genome


def _two_role_genome(**changes) -> OrgGenome:
    base = OrgGenome(
        roles=(
            RoleSpec("coder", "coder", ("Write a correct Python function.",), "small"),
            RoleSpec("reviewer", "reviewer", ("Look for bugs.", "Be concise."), "small"),
        ),
        max_retries=1,
    )
    return replace(base, **changes)


def test_diversity_defaults_to_none_and_rejects_unknown_modes():
    assert EvolutionConfig().diversity == "none"
    with pytest.raises(ValueError, match="diversity"):
        run_simulated(EvolutionConfig(population=3, generations=1, diversity="niche"))


def test_crowding_with_pareto_selection_is_rejected_with_a_clear_error():
    with pytest.raises(ValueError, match="cannot be combined with selection 'pareto'"):
        run_simulated(
            EvolutionConfig(population=3, generations=1, diversity="crowding", selection="pareto")
        )


def _fake_eval(genome: OrgGenome, fitness: float) -> Evaluation:
    return Evaluation(genome.genome_id, 0.0, 0.0, fitness, ())


def test_crowding_child_replaces_its_closest_parent_only_when_fitter():
    parent_a = seed_genome()
    parent_b = _two_role_genome()
    child_close = replace(parent_a, max_retries=2)  # one step from parent_a
    child_far = replace(parent_b, max_retries=0, roles=parent_b.roles[:1])
    population = [parent_a, parent_b]
    evaluations = {
        parent_a.genome_id: _fake_eval(parent_a, 0.5),
        parent_b.genome_id: _fake_eval(parent_b, 0.4),
        child_close.genome_id: _fake_eval(child_close, 0.6),
        child_far.genome_id: _fake_eval(child_far, 0.9),
    }
    survivors = _crowding_survivors(
        population, [child_close, child_far], [(parent_a,), (parent_b,)], evaluations
    )
    # child_close is fitter than parent_a, its closest parent: it takes parent_a's slot.
    # child_far is fitter than parent_b: it takes parent_b's slot.
    assert [g.genome_id for g in survivors] == [child_close.genome_id, child_far.genome_id]


def test_crowding_ties_keep_the_parent():
    parent = seed_genome()
    child = replace(parent, max_retries=2)
    evaluations = {
        parent.genome_id: _fake_eval(parent, 0.5),
        child.genome_id: _fake_eval(child, 0.5),
    }
    survivors = _crowding_survivors([parent], [child], [(parent,)], evaluations)
    assert survivors == [parent]


def test_crowding_pairs_a_crossover_child_with_the_closer_of_its_parents():
    near = seed_genome()
    far = _two_role_genome(max_retries=4)
    child = replace(near, max_retries=2)
    evaluations = {
        near.genome_id: _fake_eval(near, 0.2),
        far.genome_id: _fake_eval(far, 0.2),
        child.genome_id: _fake_eval(child, 0.7),
    }
    survivors = _crowding_survivors([near, far], [child], [(far, near)], evaluations)
    assert survivors == [child, far]


def test_crowding_refuses_a_child_whose_structure_is_already_present():
    parent = seed_genome()
    other = _two_role_genome()
    twin = replace(other, roles=tuple(reversed(other.roles)))  # same structure, new id
    evaluations = {
        parent.genome_id: _fake_eval(parent, 0.1),
        other.genome_id: _fake_eval(other, 0.2),
        twin.genome_id: _fake_eval(twin, 0.9),
    }
    survivors = _crowding_survivors([parent, other], [twin], [(parent,)], evaluations)
    assert survivors == [parent, other]


def test_crowding_run_keeps_population_size_and_fitness_never_drops():
    result = run_simulated(
        EvolutionConfig(population=8, generations=8, seed=0, diversity="crowding")
    )
    for record in result.generations:
        assert len(record.members) == 8
        assert record.distinct == 8
    bests = [record.members[0][1] for record in result.generations]
    for before, after in zip(bests, bests[1:], strict=False):
        assert after >= before
    for genome in result.genomes.values():
        genome.validate()


def test_crowding_run_is_deterministic_and_independent_of_workers():
    base = EvolutionConfig(population=6, generations=4, seed=2, diversity="crowding")
    serial = json.dumps(run_simulated(base).to_dict(), sort_keys=True)
    again = json.dumps(run_simulated(base).to_dict(), sort_keys=True)
    parallel = json.dumps(run_simulated(replace(base, workers=3)).to_dict(), sort_keys=True)
    assert serial == again
    assert serial == parallel


def test_crowding_works_with_crossover_children():
    config = EvolutionConfig(
        population=6, generations=3, seed=0, crossover_rate=1.0, diversity="crowding"
    )
    result = run_simulated(config)
    assert result.crossovers > 0
    assert all(record.distinct == 6 for record in result.generations)
    for genome in result.genomes.values():
        genome.validate()


def test_generation_records_carry_diversity_and_the_run_dict_shows_it():
    result = run_simulated(EvolutionConfig(population=4, generations=2, seed=0))
    data = result.to_dict()
    assert "diversity" not in data["config"]  # left out at its default
    for record, saved in zip(result.generations, data["generations"], strict=True):
        assert saved["distinct"] == record.distinct
        assert saved["mean_distance"] == pytest.approx(record.mean_distance, abs=1e-6)
        members = [gid for gid, _ in record.members]
        expected = population_diversity([result.genomes[gid] for gid in members])
        assert record.distinct == expected.distinct
        assert record.mean_distance == pytest.approx(expected.mean_distance)


def test_cli_accepts_crowding_and_saves_it_in_run_json(tmp_path):
    out = tmp_path / "crowd"
    argv = [
        "run",
        "--generations",
        "1",
        "--population",
        "3",
        "--diversity",
        "crowding",
        "--out",
        str(out),
    ]
    assert main(argv) == 0
    saved = json.loads((out / "run.json").read_text())
    assert saved["config"]["diversity"] == "crowding"
    assert len(saved["generations"]) == 2


def test_cli_rejects_crowding_with_pareto(tmp_path, capsys):
    argv = [
        "run",
        "--generations",
        "1",
        "--population",
        "3",
        "--diversity",
        "crowding",
        "--selection",
        "pareto",
        "--out",
        str(tmp_path / "bad"),
    ]
    assert main(argv) == 1
    assert "cannot be combined" in capsys.readouterr().err


def test_evolution_module_exposes_the_mode_tuple():
    assert evolution.DIVERSITY_MODES == ("none", "crowding")
