"""Compare no diversity pressure with deterministic crowding on the simulator, over several seeds.

Each seed runs twice, once with ``diversity`` "none" and once with "crowding". Both use
population 8 and 8 generations, scalar selection and mutation only, the same settings as
``compare_selection.py``. The table shows, for each arm, the mean over seeds of the champion's
held-out pass rate, its held-out mean cost and the final generation's mean pairwise genome
distance, each with a 95% percentile bootstrap interval over the seeds. The last column is the
paired difference (crowding minus none) per seed, with its interval. An interval that excludes
zero is a real difference. Held-out seeds are never seen by selection (see
``simulated_backends``).

    python scripts/compare_diversity.py            # 10 seeds
    python scripts/compare_diversity.py --seeds 2  # quick check
"""

from __future__ import annotations

import argparse
import statistics

from evorg.evolution import EvolutionConfig, RunResult, run_simulated
from evorg.stats import bootstrap_ci, paired_bootstrap_ci

MODES = ("none", "crowding")
POPULATION = 8
GENERATIONS = 8
LABEL_WIDTH = 34
CELL_WIDTH = 36


def run_all(mode: str, seeds: range) -> list[RunResult]:
    """One simulated run per seed with the given diversity mode."""
    return [
        run_simulated(
            EvolutionConfig(
                population=POPULATION, generations=GENERATIONS, seed=seed, diversity=mode
            )
        )
        for seed in seeds
    ]


def summarize(runs: list[RunResult]) -> dict[str, list[float]]:
    """Per-seed numbers for the table."""
    return {
        "held_out": [run.holdout["champion"].pass_rate for run in runs],
        "held_out_cost": [run.holdout["champion"].mean_cost_usd for run in runs],
        "final_distance": [run.generations[-1].mean_distance for run in runs],
    }


def _cell(values: list[float], spec: str) -> str:
    """Mean and 95% interval, e.g. ``0.8500 [0.8000, 0.9000]``."""
    low, high = bootstrap_ci(values)
    return f"{statistics.fmean(values):{spec}} [{low:{spec}}, {high:{spec}}]"


def _diff_cell(none: list[float], crowding: list[float], spec: str) -> str:
    """Mean of the paired differences crowding minus none, with its interval."""
    low, high = paired_bootstrap_ci(crowding, none)
    mean = statistics.fmean(b - a for a, b in zip(none, crowding, strict=True))
    return f"{mean:+{spec}} [{low:+{spec}}, {high:+{spec}}]"


def _row(label: str, none: list[float], crowding: list[float], spec: str) -> str:
    return (
        f"{label:<{LABEL_WIDTH}}{_cell(none, spec):>{CELL_WIDTH}}"
        f"{_cell(crowding, spec):>{CELL_WIDTH}}{_diff_cell(none, crowding, spec):>{CELL_WIDTH}}"
    )


def print_table(none: dict[str, list[float]], crowding: dict[str, list[float]], n: int) -> None:
    """Print means over seeds with intervals and the paired held-out wins."""
    print(
        f"{n} seeds, population {POPULATION}, generations {GENERATIONS}, scalar selection, "
        "mutation only. Means over seeds with 95% bootstrap intervals; scalar fitness = pass "
        "rate - 10 x mean cost."
    )
    print()
    header = "none", "crowding", "crowding minus none"
    print(f"{'':<{LABEL_WIDTH}}" + "".join(f"{h:>{CELL_WIDTH}}" for h in header))
    print(_row("champion held-out pass rate", none["held_out"], crowding["held_out"], ".1%"))
    print(
        f"{'  std over seeds':<{LABEL_WIDTH}}"
        f"{statistics.stdev(none['held_out']):>{CELL_WIDTH}.1%}"
        f"{statistics.stdev(crowding['held_out']):>{CELL_WIDTH}.1%}"
    )
    print(
        _row(
            "champion held-out mean cost",
            none["held_out_cost"],
            crowding["held_out_cost"],
            ".6f",
        )
    )
    print(
        _row(
            "final mean pairwise distance",
            none["final_distance"],
            crowding["final_distance"],
            ".3f",
        )
    )
    higher = sum(b > a for a, b in zip(none["held_out"], crowding["held_out"], strict=True))
    lower = sum(b < a for a, b in zip(none["held_out"], crowding["held_out"], strict=True))
    print(f"held-out, seeds where crowding scored higher / lower: {higher} / {lower}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=10, help="seeds 0 to N-1 (default 10)")
    args = parser.parse_args(argv)
    if args.seeds < 2:
        parser.error("--seeds needs at least 2 for a standard deviation")
    seeds = range(args.seeds)
    none, crowding = (summarize(run_all(mode, seeds)) for mode in MODES)
    print_table(none, crowding, args.seeds)


if __name__ == "__main__":
    main()
