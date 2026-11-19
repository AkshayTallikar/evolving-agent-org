"""Compare scalar and Pareto selection on the simulator, over several seeds.

Each seed runs twice, once with ``selection`` "scalar" (rank by fitness) and once with "pareto"
(rank by Pareto front and crowding on pass rate and cost). Both use population 8 and 8
generations. The table shows means over seeds of the champion's train pass rate, its held-out
pass rate and mean cost, and the size of the final cost/quality frontier. Held-out seeds are
never seen by selection (see ``simulated_backends``).

    python scripts/compare_selection.py            # 10 seeds
    python scripts/compare_selection.py --seeds 2  # quick check
"""

from __future__ import annotations

import argparse
import statistics

from evorg.evolution import EvolutionConfig, RunResult, run_simulated

MODES = ("scalar", "pareto")
POPULATION = 8
GENERATIONS = 8
LABEL_WIDTH = 26


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


def _line(label: str, scalar: float, pareto: float, fmt: str) -> str:
    return f"{label:<{LABEL_WIDTH}}{scalar:>14{fmt}}{pareto:>14{fmt}}"


def print_table(scalar: dict[str, list[float]], pareto: dict[str, list[float]], n: int) -> None:
    """Print means over seeds, the held-out spread and paired held-out wins."""
    print(
        f"{n} seeds, population {POPULATION}, generations {GENERATIONS}. "
        "Means over seeds; scalar fitness = pass rate - 10 x mean cost."
    )
    print()
    print(f"{'':<{LABEL_WIDTH}}{'scalar':>14}{'pareto':>14}")
    print(_line("champion train pass rate", *_means(scalar, pareto, "final_train"), ".1%"))
    print(_line("champion held-out pass rate", *_means(scalar, pareto, "held_out"), ".1%"))
    print(_line("  std over seeds", *_stdevs(scalar, pareto, "held_out"), ".1%"))
    print(_line("champion held-out mean cost", *_means(scalar, pareto, "held_out_cost"), ".6f"))
    print(_line("frontier size", *_means(scalar, pareto, "frontier"), ".1f"))
    higher = sum(b > a for a, b in zip(scalar["held_out"], pareto["held_out"], strict=True))
    lower = sum(b < a for a, b in zip(scalar["held_out"], pareto["held_out"], strict=True))
    print(f"held-out, seeds where pareto scored higher / lower: {higher} / {lower}")


def _means(scalar: dict[str, list[float]], pareto: dict[str, list[float]], key: str):
    return statistics.fmean(scalar[key]), statistics.fmean(pareto[key])


def _stdevs(scalar: dict[str, list[float]], pareto: dict[str, list[float]], key: str):
    return statistics.stdev(scalar[key]), statistics.stdev(pareto[key])


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
