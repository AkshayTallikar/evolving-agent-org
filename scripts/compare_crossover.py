"""Compare evolution with and without crossover on the simulator, over 10 seeds.

Each seed runs twice, once with ``crossover_rate`` 0.0 (mutation only) and once with 0.3,
using population 8 and 8 generations. The table shows the mean best training fitness per
generation and the means of the final numbers, including the champion's held-out pass rate
(held-out seeds are never seen by selection, see ``simulated_backends``).

    python scripts/compare_crossover.py
"""

from __future__ import annotations

import statistics

from evorg.evolution import EvolutionConfig, RunResult, run_simulated

SEEDS = range(10)
RATES = (0.0, 0.3)
POPULATION = 8
GENERATIONS = 8
LABEL_WIDTH = 26


def run_all(rate: float) -> list[RunResult]:
    """One simulated run per seed at the given crossover rate."""
    return [
        run_simulated(
            EvolutionConfig(
                population=POPULATION, generations=GENERATIONS, seed=seed, crossover_rate=rate
            )
        )
        for seed in SEEDS
    ]


def summarize(runs: list[RunResult]) -> dict[str, list[float]]:
    """Per-seed numbers: best fitness by generation, then the final scalars."""
    champions = [run.evaluations[run.champion_id] for run in runs]
    return {
        "best_by_gen": [[r.members[0][1] for r in run.generations] for run in runs],
        "final_train": [ev.pass_rate for ev in champions],
        "held_out": [run.holdout["champion"].pass_rate for run in runs],
        "evaluated": [float(len(run.evaluations)) for run in runs],
        "crossovers": [float(run.crossovers) for run in runs],
    }


def _line(label: str, off: float, on: float, fmt: str) -> str:
    return f"{label:<{LABEL_WIDTH}}{off:>14{fmt}}{on:>14{fmt}}"


def print_table(off: dict[str, list], on: dict[str, list]) -> None:
    """Print the comparison: means over seeds, plus held-out spread and paired wins."""
    print(
        f"{len(SEEDS)} seeds, population {POPULATION}, generations {GENERATIONS}. "
        "Means over seeds; fitness = pass rate - 10 x mean cost."
    )
    print()
    print(f"{'':<{LABEL_WIDTH}}{'crossover 0.0':>14}{'crossover 0.3':>14}")
    for gen in range(GENERATIONS + 1):
        a = statistics.fmean(best[gen] for best in off["best_by_gen"])
        b = statistics.fmean(best[gen] for best in on["best_by_gen"])
        print(_line(f"best train fitness, gen {gen}", a, b, ".4f"))
    print()
    print(_line("final train pass rate", *_means(off, on, "final_train"), ".1%"))
    print(_line("held-out pass rate", *_means(off, on, "held_out"), ".1%"))
    print(_line("  std over seeds", *_stdevs(off, on, "held_out"), ".1%"))
    print(_line("genomes evaluated", *_means(off, on, "evaluated"), ".1f"))
    print(_line("crossover children", *_means(off, on, "crossovers"), ".1f"))
    higher = sum(b > a for a, b in zip(off["held_out"], on["held_out"], strict=True))
    lower = sum(b < a for a, b in zip(off["held_out"], on["held_out"], strict=True))
    print(f"held-out, seeds where crossover 0.3 scored higher / lower: {higher} / {lower}")


def _means(off: dict[str, list], on: dict[str, list], key: str) -> tuple[float, float]:
    return statistics.fmean(off[key]), statistics.fmean(on[key])


def _stdevs(off: dict[str, list], on: dict[str, list], key: str) -> tuple[float, float]:
    return statistics.stdev(off[key]), statistics.stdev(on[key])


def main() -> None:
    off, on = (summarize(run_all(rate)) for rate in RATES)
    print_table(off, on)


if __name__ == "__main__":
    main()
