"""Compare evolution with and without crossover on the simulator, over 10 seeds.

Each seed runs twice, once with ``crossover_rate`` 0.0 (mutation only) and once with 0.3,
using population 8 and 8 generations. The table shows the mean best training fitness per
generation and the means of the final numbers, including the champion's held-out pass rate
(held-out seeds are never seen by selection, see ``simulated_backends``). Every mean has a
95% percentile bootstrap interval over the seeds. The last column is the paired difference
(crossover 0.3 minus 0.0) per seed, with its interval; an interval that excludes zero is a
real difference.

    python scripts/compare_crossover.py
"""

from __future__ import annotations

import statistics

from evorg.evolution import EvolutionConfig, RunResult, run_simulated
from evorg.stats import bootstrap_ci, paired_bootstrap_ci

SEEDS = range(10)
RATES = (0.0, 0.3)
POPULATION = 8
GENERATIONS = 8
LABEL_WIDTH = 30
CELL_WIDTH = 36


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


def _cell(values: list[float], spec: str) -> str:
    """Mean and 95% interval, e.g. ``0.4036 [0.3500, 0.4600]``."""
    low, high = bootstrap_ci(values)
    mean = statistics.fmean(values)
    return f"{mean:{spec}} [{low:{spec}}, {high:{spec}}]"


def _diff_cell(off: list[float], on: list[float], spec: str) -> str:
    """Mean of the paired differences on minus off, with its interval."""
    low, high = paired_bootstrap_ci(on, off)
    mean = statistics.fmean(b - a for a, b in zip(off, on, strict=True))
    return f"{mean:+{spec}} [{low:+{spec}}, {high:+{spec}}]"


def _row(label: str, off: list[float], on: list[float], spec: str, paired: bool = True) -> str:
    diff = _diff_cell(off, on, spec) if paired else ""
    return (
        f"{label:<{LABEL_WIDTH}}{_cell(off, spec):>{CELL_WIDTH}}"
        f"{_cell(on, spec):>{CELL_WIDTH}}{diff:>{CELL_WIDTH}}"
    )


def print_table(off: dict[str, list], on: dict[str, list]) -> None:
    """Print the comparison: means over seeds with intervals, paired differences, wins."""
    print(
        f"{len(SEEDS)} seeds, population {POPULATION}, generations {GENERATIONS}. "
        "Means over seeds with 95% bootstrap intervals; fitness = pass rate - 10 x mean cost."
    )
    print()
    header = "crossover 0.0", "crossover 0.3", "0.3 minus 0.0"
    print(f"{'':<{LABEL_WIDTH}}" + "".join(f"{h:>{CELL_WIDTH}}" for h in header))
    for gen in range(GENERATIONS + 1):
        a = [best[gen] for best in off["best_by_gen"]]
        b = [best[gen] for best in on["best_by_gen"]]
        print(_row(f"best train fitness, gen {gen}", a, b, ".4f", paired=False))
    print()
    print(_row("final train pass rate", off["final_train"], on["final_train"], ".1%"))
    print(_row("held-out pass rate", off["held_out"], on["held_out"], ".1%"))
    print(_line_std("  std over seeds", off["held_out"], on["held_out"]))
    print(_row("genomes evaluated", off["evaluated"], on["evaluated"], ".1f"))
    print(_row("crossover children", off["crossovers"], on["crossovers"], ".1f"))
    higher = sum(b > a for a, b in zip(off["held_out"], on["held_out"], strict=True))
    lower = sum(b < a for a, b in zip(off["held_out"], on["held_out"], strict=True))
    print(f"held-out, seeds where crossover 0.3 scored higher / lower: {higher} / {lower}")


def _line_std(label: str, off: list[float], on: list[float]) -> str:
    return (
        f"{label:<{LABEL_WIDTH}}{statistics.stdev(off):>{CELL_WIDTH}.1%}"
        f"{statistics.stdev(on):>{CELL_WIDTH}.1%}"
    )


def main() -> None:
    off, on = (summarize(run_all(rate)) for rate in RATES)
    print_table(off, on)


if __name__ == "__main__":
    main()
