import json
import random
from types import SimpleNamespace

import pytest

from evorg import evolution
from evorg.bench import load_benchmark, task_index
from evorg.evolution import (
    Evaluation,
    EvolutionConfig,
    evaluate,
    evaluate_many,
    evolve,
    run_simulated,
    selection_keys,
    simulated_backends,
)
from evorg.executor import TaskResult, run_task
from evorg.genome import seed_genome
from evorg.llm.simulated import SimulatedBackend
from evorg.mutation import Architect

SMALL = EvolutionConfig(population=4, generations=2, seed=0)


def _result(task_id: str, passed: bool, fraction: float, attempts: int, cost: float) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        passed=passed,
        pass_fraction=fraction,
        attempts=attempts,
        calls=attempts + 1,
        input_tokens=100,
        output_tokens=50,
        cost_usd=cost,
        transcript=(),
    )


def test_evaluate_matches_task_results():
    backend = SimulatedBackend(task_index(), seed=0)
    genome = seed_genome()
    tasks = load_benchmark("train")[:4]
    cost_weight = 10.0

    ev = evaluate(genome, tasks, backend, cost_weight)
    expected = [run_task(genome, task, backend) for task in tasks]

    assert ev.genome_id == genome.genome_id
    assert ev.results == tuple(expected)
    pass_rate = sum(r.passed for r in expected) / len(expected)
    mean_cost = sum(r.cost_usd for r in expected) / len(expected)
    assert ev.pass_rate == pytest.approx(pass_rate)
    assert ev.mean_cost_usd == pytest.approx(mean_cost)
    assert ev.fitness == pytest.approx(pass_rate - cost_weight * mean_cost)


def test_evaluate_many_with_one_backend_equals_evaluate():
    backend = SimulatedBackend(task_index(), seed=0)
    genome = seed_genome()
    tasks = load_benchmark("train")[:4]

    assert evaluate_many(genome, tasks, [backend], 10.0) == evaluate(genome, tasks, backend, 10.0)


def test_evaluate_many_averages_scores_over_backends():
    index = task_index()
    backends = [SimulatedBackend(index, seed=0), SimulatedBackend(index, seed=1000)]
    genome = seed_genome()
    tasks = load_benchmark("train")[:4]
    cost_weight = 10.0

    many = evaluate_many(genome, tasks, backends, cost_weight)
    singles = [evaluate(genome, tasks, b, cost_weight) for b in backends]

    assert many.genome_id == genome.genome_id
    assert many.pass_rate == pytest.approx(sum(e.pass_rate for e in singles) / 2)
    assert many.mean_cost_usd == pytest.approx(sum(e.mean_cost_usd for e in singles) / 2)
    assert many.fitness == pytest.approx(sum(e.fitness for e in singles) / 2)
    assert many.results == singles[0].results + singles[1].results
    assert many.report().startswith("Pass rate ")


def test_evaluate_many_needs_a_backend():
    with pytest.raises(ValueError):
        evaluate_many(seed_genome(), load_benchmark("train")[:1], [], 10.0)


def test_evaluate_with_no_tasks_scores_zero():
    backend = SimulatedBackend(task_index(), seed=0)
    ev = evaluate(seed_genome(), [], backend, 10.0)
    assert (ev.pass_rate, ev.mean_cost_usd, ev.fitness) == (0.0, 0.0, 0.0)


def test_report_lists_failures_and_stays_short():
    results = tuple(
        _result(f"task_{i}", passed=False, fraction=0.25, attempts=2, cost=0.0004)
        for i in range(200)
    )
    ev = Evaluation("abc", 0.0, 0.0004, -0.004, results)
    text = ev.report()
    assert text.startswith("Pass rate 0/200 (0%), mean cost $0.00040 per task.")
    assert "- task_0: 25% of tests passed, 2 attempts" in text
    assert len(text) <= 1500
    assert "more" in text


def test_report_on_all_passing_has_no_failure_lines():
    results = (_result("a", True, 1.0, 1, 0.001), _result("b", True, 1.0, 1, 0.003))
    ev = Evaluation("abc", 1.0, 0.002, 0.98, results)
    assert ev.report() == "Pass rate 2/2 (100%), mean cost $0.00200 per task."


def test_run_is_deterministic_for_same_seed():
    config = EvolutionConfig(population=4, generations=2, seed=0)
    first = json.dumps(run_simulated(config).to_dict(), sort_keys=True)
    second = json.dumps(run_simulated(config).to_dict(), sort_keys=True)
    assert first == second


def test_different_seed_gives_different_run():
    a = run_simulated(EvolutionConfig(population=4, generations=2, seed=0))
    b = run_simulated(EvolutionConfig(population=4, generations=2, seed=1))
    assert json.dumps(a.to_dict(), sort_keys=True) != json.dumps(b.to_dict(), sort_keys=True)


def test_best_fitness_never_drops_between_generations():
    result = run_simulated(EvolutionConfig(population=6, generations=4, seed=0))
    bests = [record.members[0][1] for record in result.generations]
    assert len(bests) == 5
    for before, after in zip(bests, bests[1:], strict=False):
        assert after >= before


def test_every_genome_validates_and_champion_lineage_is_a_chain():
    result = run_simulated(EvolutionConfig(population=6, generations=4, seed=0))
    for genome in result.genomes.values():
        genome.validate()

    chain = result.lineage(result.champion_id)
    assert chain[0].genome_id == result.seed_id
    assert chain[-1].genome_id == result.champion_id
    assert chain[0].parent_ids == ()
    for previous, step in zip(chain, chain[1:], strict=False):
        assert step.parent_ids[0] == previous.genome_id
        assert step.generation == previous.generation + 1


def test_no_genome_is_evaluated_twice(monkeypatch):
    evaluated: list[str] = []
    real_evaluate = evolution.evaluate

    def counting_evaluate(genome, tasks, backend, cost_weight):
        if tasks and tasks[0].split == "train":
            evaluated.append(genome.genome_id)
        return real_evaluate(genome, tasks, backend, cost_weight)

    monkeypatch.setattr(evolution, "evaluate", counting_evaluate)
    result = run_simulated(EvolutionConfig(population=6, generations=4, seed=0))

    assert len(evaluated) == len(set(evaluated))
    assert set(evaluated) == set(result.evaluations) == set(result.genomes)


def test_holdout_reports_champion_and_seed_on_four_tasks():
    result = run_simulated(SMALL)
    assert set(result.holdout) == {"champion", "seed"}
    assert len(result.holdout["champion"].results) == 4
    assert len(result.holdout["seed"].results) == 4
    assert result.holdout["seed"].genome_id == result.seed_id
    assert result.holdout["champion"].genome_id == result.champion_id


def test_to_dict_is_json_serialisable():
    result = run_simulated(SMALL)
    data = result.to_dict()
    text = json.dumps(data)
    assert json.loads(text)["champion_id"] == result.champion_id
    assert set(data["holdout"]) == {"champion", "seed"}
    assert "transcript" not in text


def test_evolution_improves_on_the_seed():
    result = run_simulated(EvolutionConfig(population=6, generations=4, seed=0))
    champion = result.evaluations[result.champion_id].fitness
    seed = result.evaluations[result.seed_id].fitness
    assert champion > seed


def test_run_with_two_eval_seeds_is_deterministic():
    config = EvolutionConfig(population=4, generations=2, seed=0, eval_seeds=2)
    first = json.dumps(run_simulated(config).to_dict(), sort_keys=True)
    second = json.dumps(run_simulated(config).to_dict(), sort_keys=True)
    assert first == second
    assert json.loads(first)["config"]["eval_seeds"] == 2


def test_simulated_backends_use_disjoint_seeds_for_holdout():
    config = EvolutionConfig(seed=7, eval_seeds=3)
    train, holdout = simulated_backends(config)
    assert [b.seed for b in train] == [7, 1007, 2007]
    assert [b.seed for b in holdout] == [500_007, 501_007, 502_007]
    assert not set(b.seed for b in train) & set(b.seed for b in holdout)


def test_holdout_still_gets_a_seed_when_eval_seeds_is_one():
    _, holdout = simulated_backends(EvolutionConfig(seed=0, eval_seeds=1))
    assert [b.seed for b in holdout] == [500_000]


def test_eval_seeds_below_one_is_rejected():
    backend = SimulatedBackend(task_index(), seed=0)
    architect = Architect(backend, random.Random(1))
    with pytest.raises(ValueError, match="eval_seeds"):
        evolve(EvolutionConfig(eval_seeds=0), backend, [], [], architect)
    with pytest.raises(ValueError, match="eval_seeds"):
        run_simulated(EvolutionConfig(eval_seeds=0))


def test_invalid_config_is_rejected():
    backend = SimulatedBackend(task_index(), seed=0)
    architect = Architect(backend, random.Random(1))
    with pytest.raises(ValueError):
        evolve(EvolutionConfig(population=0), backend, [], [], architect)
    with pytest.raises(ValueError):
        evolve(EvolutionConfig(population=2, elite=3), backend, [], [], architect)


@pytest.mark.parametrize("rate", [-0.1, 1.5, float("nan")])
def test_crossover_rate_outside_zero_to_one_is_rejected(rate):
    with pytest.raises(ValueError, match="crossover_rate"):
        run_simulated(EvolutionConfig(population=4, generations=1, crossover_rate=rate))


def test_crossover_rate_zero_is_the_default_and_pinned_champion_is_unchanged():
    assert EvolutionConfig().crossover_rate == 0.0
    result = run_simulated(EvolutionConfig(population=8, generations=8, seed=0))
    assert result.champion_id == "fba3bf8545"
    assert result.evaluations[result.champion_id].fitness == pytest.approx(0.624522, abs=1e-6)
    assert result.crossovers == 0


def test_rate_one_breeds_children_from_two_recorded_parents():
    config = EvolutionConfig(population=6, generations=3, seed=0, crossover_rate=1.0)
    result = run_simulated(config)
    for genome in result.genomes.values():
        genome.validate()
    two_parent = [g for g in result.genomes.values() if len(g.parent_ids) == 2]
    assert two_parent
    for genome in two_parent:
        assert all(parent in result.genomes for parent in genome.parent_ids)
        assert genome.mutation_note.startswith("crossover of ")
        assert genome.generation >= 1


def test_crossover_count_is_recorded_in_the_result_and_dict():
    config = EvolutionConfig(population=6, generations=3, seed=0, crossover_rate=1.0)
    result = run_simulated(config)
    assert 0 < result.crossovers <= 6 * 3
    assert result.to_dict()["crossovers"] == result.crossovers


def test_run_with_crossover_is_deterministic():
    config = EvolutionConfig(population=6, generations=3, seed=0, crossover_rate=0.3)
    first = json.dumps(run_simulated(config).to_dict(), sort_keys=True)
    second = json.dumps(run_simulated(config).to_dict(), sort_keys=True)
    assert first == second
    assert json.loads(first)["config"]["crossover_rate"] == 0.3


def test_crossover_rate_zero_and_omitted_give_the_same_run():
    explicit = run_simulated(
        EvolutionConfig(population=4, generations=2, seed=3, crossover_rate=0)
    )
    omitted = run_simulated(EvolutionConfig(population=4, generations=2, seed=3))
    assert explicit.to_dict() == omitted.to_dict()


def test_selection_defaults_to_scalar():
    assert EvolutionConfig().selection == "scalar"


@pytest.mark.parametrize("mode", ["fitness", "", "Pareto"])
def test_unknown_selection_is_rejected(mode):
    with pytest.raises(ValueError, match="selection"):
        run_simulated(EvolutionConfig(population=4, generations=1, selection=mode))


def _fake(genome_id: str, pass_rate: float, cost: float) -> Evaluation:
    fitness = pass_rate - 10 * cost
    return Evaluation(genome_id, pass_rate, cost, fitness, ())


def _genome(genome_id: str):
    return SimpleNamespace(genome_id=genome_id)  # selection_keys only reads genome_id


def test_pareto_keys_put_front_zero_first_and_scalar_keys_follow_fitness():
    evals = {
        "a": _fake("a", 0.9, 0.02),  # front 0
        "b": _fake("b", 0.5, 0.01),  # front 0, cheapest
        "c": _fake("c", 0.4, 0.05),  # dominated by a and b
    }
    members = [_genome(k) for k in ("c", "a", "b")]
    keys = selection_keys(members, evals, "pareto")
    order = sorted(members, key=lambda g: keys[g.genome_id])
    assert [g.genome_id for g in order][-1] == "c"
    assert {g.genome_id for g in order[:2]} == {"a", "b"}

    scalar = selection_keys(members, evals, "scalar")
    assert scalar["a"] == evolution._rank_key(evals["a"])


def test_pareto_run_is_deterministic_for_a_seed():
    config = EvolutionConfig(population=5, generations=2, seed=1, selection="pareto")
    first = json.dumps(run_simulated(config).to_dict(), sort_keys=True)
    second = json.dumps(run_simulated(config).to_dict(), sort_keys=True)
    assert first == second


def test_pareto_run_keeps_the_fitness_champion_and_a_non_dominated_frontier():
    result = run_simulated(EvolutionConfig(population=6, generations=3, seed=0, selection="pareto"))
    best_fitness = max(ev.fitness for ev in result.evaluations.values())
    assert result.evaluations[result.champion_id].fitness == best_fitness

    points = {
        gid: (ev.pass_rate, ev.mean_cost_usd) for gid, ev in result.evaluations.items()
    }
    assert result.frontier
    assert set(result.frontier) <= set(points)
    for gid in result.frontier:
        rate, cost = points[gid]
        for other_rate, other_cost in points.values():
            dominated = other_rate >= rate and other_cost <= cost and (
                other_rate > rate or other_cost < cost
            )
            assert not dominated
    costs = [points[gid][1] for gid in result.frontier]
    assert costs == sorted(costs)


def test_frontier_is_in_to_dict_in_both_modes():
    for mode in ("scalar", "pareto"):
        data = run_simulated(EvolutionConfig(population=4, generations=1, selection=mode)).to_dict()
        assert data["frontier"]
        assert data["config"]["selection"] == mode
        json.dumps(data)
