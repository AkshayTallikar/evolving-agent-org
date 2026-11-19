"""Compare scalar and Pareto selection on the simulator, over several seeds.

Each seed runs twice, once with ``selection`` "scalar" (rank by fitness) and once with "pareto"
(rank by Pareto front and crowding on pass rate and cost). Both use population 8 and 8
generations. The table shows means over seeds of the champion's train pass rate, its held-out
pass rate and mean cost, and the size of the final cost/quality frontier. Every mean has a 95%
percentile bootstrap interval over the seeds. The last column is the paired difference
(pareto minus scalar) per seed, with its interval; an interval that excludes zero is a real
difference. Held-out seeds are never seen by selection (see ``simulated_backends``).

    python scripts/compare_selection.py            # 10 seeds
    python scripts/compare_selection.py --seeds 2  # quick check
"""

from __future__ import annotations

import argparse
import statistics

from evorg.evolution import EvolutionConfig, RunResult, run_simulated
from evorg.stats import bootstrap_ci, paired_bootstrap_ci

MODES = ("scalar", "pareto")
POPULATION = 8
GENERATIONS = 8
LABEL_WIDTH = 30
CELL_WIDTH = 36


def run_all(mode: str, seeds: range) -> list[RunResult]:
    """One simulated run per seed with the given selection mode."""
    return [
        run_simulated(
            EvolutionConfig(
                population=POPULATION, generations=GENERATIONS, seed=seed, selection=mode
            )
        )
        for seed in seeds
    ]


def summarize(runs: list[RunResult]) -> dict[str, list[float]]:
    """Per-seed numbers for the table."""
    champions = [run.evaluations[run.champion_id] for run in runs]
    return {
        "final_train": [ev.pass_rate for ev in champions],
        "held_out": [run.holdout["champion"].pass_rate for run in runs],
        "held_out_cost": [run.holdout["champion"].mean_cost_usd for run in runs],
        "frontier": [float(len(run.frontier)) for run in runs],
    }


def _cell(values: list[float], spec: str) -> str:
    """Mean and 95% interval, e.g. ``0.8500 [0.8000, 0.9000]``."""
    low, high = bootstrap_ci(values)
    return f"{statistics.fmean(values):{spec}} [{low:{spec}}, {high:{spec}}]"


def _diff_cell(scalar: list[float], pareto: list[float], spec: str) -> str:
    """Mean of the paired differences pareto minus scalar, with its interval."""
    low, high = paired_bootstrap_ci(pareto, scalar)
    mean = statistics.fmean(b - a for a, b in zip(scalar, pareto, strict=True))
    return f"{mean:+{spec}} [{low:+{spec}}, {high:+{spec}}]"


def _row(label: str, scalar: list[float], pareto: list[float], spec: str) -> str:
    return (
        f"{label:<{LABEL_WIDTH}}{_cell(scalar, spec):>{CELL_WIDTH}}"
        f"{_cell(pareto, spec):>{CELL_WIDTH}}{_diff_cell(scalar, pareto, spec):>{CELL_WIDTH}}"
    )


def print_table(scalar: dict[str, list[float]], pareto: dict[str, list[float]], n: int) -> None:
    """Print means over seeds with intervals, the held-out spread and paired held-out wins."""
    print(
        f"{n} seeds, population {POPULATION}, generations {GENERATIONS}. "
        "Means over seeds with 95% bootstrap intervals; scalar fitness = pass rate - 10 x mean "
        "cost."
    )
    print()
    header = "scalar", "pareto", "pareto minus scalar"
    print(f"{'':<{LABEL_WIDTH}}" + "".join(f"{h:>{CELL_WIDTH}}" for h in header))
    print(_row("champion train pass rate", scalar["final_train"], pareto["final_train"], ".1%"))
    print(_row("champion held-out pass rate", scalar["held_out"], pareto["held_out"], ".1%"))
    print(
        f"{'  std over seeds':<{LABEL_WIDTH}}"
        f"{statistics.stdev(scalar['held_out']):>{CELL_WIDTH}.1%}"
        f"{statistics.stdev(pareto['held_out']):>{CELL_WIDTH}.1%}"
    )
    print(
        _row(
            "champion held-out mean cost",
            scalar["held_out_cost"],
            pareto["held_out_cost"],
            ".6f",
        )
    )
    print(_row("frontier size", scalar["frontier"], pareto["frontier"], ".1f"))
    higher = sum(b > a for a, b in zip(scalar["held_out"], pareto["held_out"], strict=True))
    lower = sum(b < a for a, b in zip(scalar["held_out"], pareto["held_out"], strict=True))
    print(f"held-out, seeds where pareto scored higher / lower: {higher} / {lower}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=10, help="seeds 0 to N-1 (default 10)")
    args = parser.parse_args(argv)
    if args.seeds < 2:
        parser.error("--seeds needs at least 2 for a standard deviation")
    seeds = range(args.seeds)
    scalar, pareto = (summarize(run_all(mode, seeds)) for mode in MODES)
    print_table(scalar, pareto, args.seeds)


if __name__ == "__main__":
    main()
