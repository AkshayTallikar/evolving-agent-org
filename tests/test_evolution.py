import json
import random

import pytest

from evorg import evolution
from evorg.bench import load_benchmark, task_index
from evorg.evolution import (
    Evaluation,
    EvolutionConfig,
    evaluate,
    evolve,
    run_simulated,
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


def test_invalid_config_is_rejected():
    backend = SimulatedBackend(task_index(), seed=0)
    architect = Architect(backend, random.Random(1))
    with pytest.raises(ValueError):
        evolve(EvolutionConfig(population=0), backend, [], [], architect)
    with pytest.raises(ValueError):
        evolve(EvolutionConfig(population=2, elite=3), backend, [], [], architect)
