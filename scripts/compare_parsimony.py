"""Compare evolution with and without a parsimony penalty on the simulator, over 10 seeds.

Each seed runs twice, once with ``complexity_weight`` 0 (no penalty) and once with W = 0.01,
using population 8 and 8 generations, the same settings as compare_selection.py. The penalty
is W times the genome size (instructions over all roles plus extra roles), so at W = 0.01 one
extra instruction or role costs as much fitness as one point of pass rate, the scale the
simulator's own instruction effects are measured on.

The table shows, for each arm, the mean over seeds of the champion's held-out pass rate, its
held-out mean cost per task, its genome size and the share of champions that carry a neutral
instruction (an instruction from the gene pool with zero weight in the simulator). Every mean
has a 95% percentile bootstrap interval over the seeds. The last column is the paired
difference (penalty minus none) per seed, with its interval; an interval that excludes zero is
a real difference. Held-out seeds are never seen by selection (see ``simulated_backends``).

    python scripts/compare_parsimony.py            # 10 seeds
    python scripts/compare_parsimony.py --seeds 2  # quick check
"""

from __future__ import annotations

import argparse
import statistics

from evorg.evolution import EvolutionConfig, RunResult, run_simulated
from evorg.genes import INSTRUCTION_POOL
from evorg.genome import OrgGenome
from evorg.llm.simulated import FEATURE_WEIGHTS
from evorg.stats import bootstrap_ci, paired_bootstrap_ci

WEIGHTS = (0.0, 0.01)  # no penalty, then the penalty used in the README
POPULATION = 8
GENERATIONS = 8
LABEL_WIDTH = 34
CELL_WIDTH = 36


def neutral_instructions() -> frozenset[str]:
    """Pool instructions that no simulator keyword with a non-zero weight appears in."""
    return frozenset(
        text
        for text in INSTRUCTION_POOL
        if not any(
            weight != 0 and keyword in text.lower() for keyword, weight in FEATURE_WEIGHTS.items()
        )
    )


NEUTRAL = neutral_instructions()


def carries_neutral(genome: OrgGenome) -> bool:
    return any(text in NEUTRAL for role in genome.roles for text in role.instructions)


def run_all(weight: float, seeds: range) -> list[RunResult]:
    """One simulated run per seed with the given complexity weight."""
    return [
        run_simulated(
            EvolutionConfig(
                population=POPULATION,
                generations=GENERATIONS,
                seed=seed,
                complexity_weight=weight,
            )
        )
        for seed in seeds
    ]


def summarize(runs: list[RunResult]) -> dict[str, list[float]]:
    """Per-seed numbers for the table."""
    champions = [run.genomes[run.champion_id] for run in runs]
    return {
        "held_out": [run.holdout["champion"].pass_rate for run in runs],
        "held_out_cost": [run.holdout["champion"].mean_cost_usd for run in runs],
        "size": [float(genome.size()) for genome in champions],
        "neutral": [1.0 if carries_neutral(genome) else 0.0 for genome in champions],
    }


def _cell(values: list[float], spec: str) -> str:
    """Mean and 95% interval, e.g. ``0.8500 [0.8000, 0.9000]``."""
    low, high = bootstrap_ci(values)
    return f"{statistics.fmean(values):{spec}} [{low:{spec}}, {high:{spec}}]"


def _diff_cell(base: list[float], penalised: list[float], spec: str) -> str:
    """Mean of the paired differences penalty minus none, with its interval."""
    low, high = paired_bootstrap_ci(penalised, base)
    mean = statistics.fmean(b - a for a, b in zip(base, penalised, strict=True))
    return f"{mean:+{spec}} [{low:+{spec}}, {high:+{spec}}]"


def _row(label: str, base: list[float], penalised: list[float], spec: str) -> str:
    return (
        f"{label:<{LABEL_WIDTH}}{_cell(base, spec):>{CELL_WIDTH}}"
        f"{_cell(penalised, spec):>{CELL_WIDTH}}{_diff_cell(base, penalised, spec):>{CELL_WIDTH}}"
    )


def print_table(base: dict[str, list[float]], penalised: dict[str, list[float]], n: int) -> None:
    """Print means over seeds with intervals, then held-out wins and losses."""
    print(
        f"{n} seeds, population {POPULATION}, generations {GENERATIONS}. Means over seeds with "
        "95% bootstrap intervals; fitness = pass rate - 10 x mean cost - W x genome size."
    )
    print(f"neutral instructions: {', '.join(repr(text) for text in sorted(NEUTRAL))}")
    print()
    header = f"W = {WEIGHTS[0]:g}", f"W = {WEIGHTS[1]:g}", f"W={WEIGHTS[1]:g} minus W=0"
    print(f"{'':<{LABEL_WIDTH}}" + "".join(f"{h:>{CELL_WIDTH}}" for h in header))
    print(_row("champion held-out pass rate", base["held_out"], penalised["held_out"], ".1%"))
    print(
        _row(
            "champion held-out mean cost",
            base["held_out_cost"],
            penalised["held_out_cost"],
            ".6f",
        )
    )
    print(_row("champion genome size", base["size"], penalised["size"], ".2f"))
    print(_row("champions with a neutral instr.", base["neutral"], penalised["neutral"], ".1%"))
    higher = sum(b > a for a, b in zip(base["held_out"], penalised["held_out"], strict=True))
    lower = sum(b < a for a, b in zip(base["held_out"], penalised["held_out"], strict=True))
    print(f"held-out, seeds where the penalty scored higher / lower: {higher} / {lower}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=10, help="seeds 0 to N-1 (default 10)")
    args = parser.parse_args(argv)
    if args.seeds < 2:
        parser.error("--seeds needs at least 2 for a standard deviation")
    seeds = range(args.seeds)
    base, penalised = (summarize(run_all(weight, seeds)) for weight in WEIGHTS)
    print_table(base, penalised, args.seeds)


if __name__ == "__main__":
    main()
