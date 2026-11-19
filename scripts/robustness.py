"""Re-score evolved champions under simulator noise they never saw.

Each run evolves with one simulator seed, so a champion can partly fit that seed's
luck. This script runs evolution for 10 seeds, then scores each champion and the
seed genome on 20 fresh simulator seeds, on both the train and held-out tasks.

Every mean is printed with a 95% percentile bootstrap interval. The seed genome's interval
comes from its 20 scoring seeds. The champion intervals come from the 10 runs. The
champion-minus-seed row is a paired difference per run, against the seed genome's mean.

    python scripts/robustness.py                 # one scoring seed per genome
    python scripts/robustness.py --eval-seeds 3  # selection averages 3 seeds
"""

from __future__ import annotations

import argparse
import statistics

from evorg.bench import load_benchmark, task_index
from evorg.evolution import EvolutionConfig, evaluate, run_simulated
from evorg.genome import OrgGenome, seed_genome
from evorg.llm.simulated import SimulatedBackend
from evorg.stats import bootstrap_ci, paired_bootstrap_ci

RUN_SEEDS = range(10)
FRESH_SEEDS = range(100, 120)


def pass_rates(genome: OrgGenome, tasks, seeds) -> list[float]:
    """The genome's pass rate on each simulator seed, one value per seed."""
    return [
        evaluate(
            genome, tasks, SimulatedBackend(task_index(), seed=s), EvolutionConfig().cost_weight
        ).pass_rate
        for s in seeds
    ]


def fmt_ci(values: list[float]) -> str:
    """Mean with its 95% interval, as percentages."""
    low, high = bootstrap_ci(values)
    return f"{statistics.fmean(values):6.1%} [{low:6.1%}, {high:6.1%}]"


def fmt_diff(a: list[float], b: list[float]) -> str:
    """Mean of the paired differences a[i] - b[i] with its 95% interval, signed."""
    low, high = paired_bootstrap_ci(a, b)
    mean = statistics.fmean(x - y for x, y in zip(a, b, strict=True))
    return f"{mean:+7.1%} [{low:+7.1%}, {high:+7.1%}]"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-seeds", type=int, default=1)
    eval_seeds = parser.parse_args(argv).eval_seeds
    train, holdout = load_benchmark("train"), load_benchmark("holdout")
    in_run = []
    champ_train, champ_holdout = [], []  # one value per run: mean over the fresh seeds
    for s in RUN_SEEDS:
        config = EvolutionConfig(population=8, generations=8, seed=s, eval_seeds=eval_seeds)
        result = run_simulated(config)
        champion = result.genomes[result.champion_id]
        in_run.append(result.evaluations[result.champion_id].pass_rate)
        champ_train.append(statistics.fmean(pass_rates(champion, train, FRESH_SEEDS)))
        champ_holdout.append(statistics.fmean(pass_rates(champion, holdout, FRESH_SEEDS)))
        print(f"run seed {s}: champion {result.champion_id}")

    seed = seed_genome()
    seed_train = pass_rates(seed, train, FRESH_SEEDS)
    seed_holdout = pass_rates(seed, holdout, FRESH_SEEDS)
    print()
    print(f"eval seeds {eval_seeds}, 95% bootstrap intervals in brackets")
    print(f"{'':<26}{'train':>24}{'held-out':>24}")
    print(f"{'seed genome, fresh noise':<26}{fmt_ci(seed_train):>24}{fmt_ci(seed_holdout):>24}")
    print(f"{'champions, own run':<26}{fmt_ci(in_run):>24}{'-':>24}")
    print(f"{'champions, fresh noise':<26}{fmt_ci(champ_train):>24}{fmt_ci(champ_holdout):>24}")
    print(
        f"{'champion minus seed':<26}"
        f"{fmt_diff(champ_train, [statistics.fmean(seed_train)] * len(champ_train)):>24}"
        f"{fmt_diff(champ_holdout, [statistics.fmean(seed_holdout)] * len(champ_holdout)):>24}"
    )


if __name__ == "__main__":
    main()
