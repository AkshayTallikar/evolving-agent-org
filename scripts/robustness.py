"""Re-score evolved champions under simulator noise they never saw.

Each run evolves with one simulator seed, so a champion can partly fit that seed's
luck. This script runs evolution for 10 seeds, then scores each champion and the
seed genome on 20 fresh simulator seeds, on both the train and held-out tasks.

    python scripts/robustness.py
"""

from __future__ import annotations

from evorg.bench import load_benchmark, task_index
from evorg.evolution import EvolutionConfig, evaluate, run_simulated
from evorg.genome import OrgGenome, seed_genome
from evorg.llm.simulated import SimulatedBackend

RUN_SEEDS = range(10)
FRESH_SEEDS = range(100, 120)


def mean_pass_rate(genome: OrgGenome, tasks, seeds) -> float:
    total = 0.0
    for s in seeds:
        backend = SimulatedBackend(task_index(), seed=s)
        total += evaluate(genome, tasks, backend, EvolutionConfig().cost_weight).pass_rate
    return total / len(seeds)


def main() -> None:
    train, holdout = load_benchmark("train"), load_benchmark("holdout")
    in_run, fresh_train, fresh_holdout = [], [], []
    for s in RUN_SEEDS:
        result = run_simulated(EvolutionConfig(population=8, generations=8, seed=s))
        champion = result.genomes[result.champion_id]
        in_run.append(result.evaluations[result.champion_id].pass_rate)
        fresh_train.append(mean_pass_rate(champion, train, FRESH_SEEDS))
        fresh_holdout.append(mean_pass_rate(champion, holdout, FRESH_SEEDS))
        print(f"run seed {s}: champion {result.champion_id}")

    def avg(xs):
        return sum(xs) / len(xs)

    seed = seed_genome()
    print()
    print("                         train   held-out")
    print(f"seed genome, fresh noise {mean_pass_rate(seed, train, FRESH_SEEDS):6.1%}"
          f"  {mean_pass_rate(seed, holdout, FRESH_SEEDS):6.1%}")
    print(f"champions, own run       {avg(in_run):6.1%}       -")
    print(f"champions, fresh noise   {avg(fresh_train):6.1%}  {avg(fresh_holdout):6.1%}")


if __name__ == "__main__":
    main()
