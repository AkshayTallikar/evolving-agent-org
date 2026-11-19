"""Markdown report for a finished evolution run (design section 4.8).

``render_markdown`` only reads a ``RunResult``; it makes no calls and touches no files, so the
same run always produces the same text.
"""

from __future__ import annotations

from evorg.evolution import RunResult
from evorg.genome import OrgGenome, diff_genomes

SPARK_BLOCKS = "▁▂▃▄▅▆▇█"
SIM_NOTE = (
    "Results come from the simulated backend's hand-written capability model and show that "
    "selection works, not how real models behave."
)
NO_NOTE = "no mutation note recorded"


def sparkline(values: list[float]) -> str:
    """Scale ``values`` onto eight block characters between their min and max.

    An empty list gives an empty string. If every value is equal there is no range to scale,
    so the middle block is repeated once per value.
    """
    if not values:
        return ""
    low, high = min(values), max(values)
    if high == low:
        return SPARK_BLOCKS[len(SPARK_BLOCKS) // 2] * len(values)
    top = len(SPARK_BLOCKS) - 1
    return "".join(SPARK_BLOCKS[round((v - low) / (high - low) * top)] for v in values)


def render_markdown(run: RunResult) -> str:
    """Render a run as a markdown report with a fixed set of sections."""
    cfg = run.config
    valid = run.architect_stats.get("valid", 0)
    proposed = run.architect_stats.get("proposed", 0)
    champion = run.genomes[run.champion_id]
    champion_ev = run.evaluations[run.champion_id]
    seed = run.genomes[run.seed_id]

    seeds_note = f"scored over {cfg.eval_seeds} simulator seeds, " if cfg.eval_seeds > 1 else ""
    crossover_note = f"crossover rate {cfg.crossover_rate:g}, " if cfg.crossover_rate > 0 else ""
    selection_note = "pareto selection, " if cfg.selection == "pareto" else ""
    lines: list[str] = [
        "# evorg run",
        "",
        (
            f"Backend {run.backend}, seed {cfg.seed}, population {cfg.population}, "
            f"generations {cfg.generations}, {seeds_note}{crossover_note}{selection_note}"
            f"{len(run.evaluations)} genomes "
            f"evaluated, architect proposals valid {valid}/proposed {proposed}."
        ),
        "",
        "## Fitness by generation",
        "",
        "| generation | best fitness | best pass rate | mean fitness of population "
        "| best genome id |",
        "| --- | --- | --- | --- | --- |",
    ]
    best_fitness: list[float] = []
    for record in run.generations:
        best_id, fitness = record.members[0]
        best_ev = run.evaluations[best_id]
        mean = sum(f for _, f in record.members) / len(record.members)
        best_fitness.append(fitness)
        lines.append(
            f"| {record.index} | {fitness:.4f} | {best_ev.pass_rate:.1%} | {mean:.4f} | {best_id} |"
        )
    lines += ["", f"Best fitness: {sparkline(best_fitness)}", ""]

    lines += [
        "## Champion",
        "",
        (
            f"Genome {run.champion_id}, train pass rate {champion_ev.pass_rate:.1%}, "
            f"mean cost per task ${champion_ev.mean_cost_usd:.6f}."
        ),
        "",
    ]
    for role in champion.roles:
        lines.append(f"- **{role.name}** ({role.kind}, {role.model})")
        lines.extend(f"  - {text}" for text in role.instructions)
    lines += ["", f"Retry budget: {champion.max_retries}", ""]

    lines += ["## What changed from the seed", ""]
    changes = diff_genomes(seed, champion)
    if changes:
        lines.extend(f"- {line}" for line in changes)
    else:
        lines.append("No change: the seed was never beaten.")
    lines.append("")

    lines += ["## Lineage", ""]
    for i, genome in enumerate(run.lineage(run.champion_id), start=1):
        lines.append(_lineage_line(run, i, genome))
    lines.append("")

    lines += [
        "## Cost/quality frontier",
        "",
        "Genomes no other evaluated genome beats on both train pass rate and mean cost, "
        "cheapest first.",
        "",
        "| genome | pass rate | mean cost per task |",
        "| --- | --- | --- |",
    ]
    for genome_id in run.frontier:
        ev = run.evaluations[genome_id]
        lines.append(f"| {genome_id} | {ev.pass_rate:.1%} | ${ev.mean_cost_usd:.6f} |")
    lines.append("")

    lines += [
        "## Held-out tasks",
        "",
        "| genome | pass rate | mean cost per task |",
        "| --- | --- | --- |",
    ]
    for name in ("seed", "champion"):
        ev = run.holdout[name]
        lines.append(f"| {name} | {ev.pass_rate:.1%} | ${ev.mean_cost_usd:.6f} |")
    lines += ["", "Held-out tasks were never used for selection.", ""]

    if run.backend == "sim":
        lines += ["Held-out tasks were scored on simulator seeds that selection never saw.", ""]
        lines += [SIM_NOTE, ""]
    return "\n".join(lines)


def _lineage_line(run: RunResult, index: int, genome: OrgGenome) -> str:
    fitness = run.evaluations[genome.genome_id].fitness
    note = "seed" if genome.genome_id == run.seed_id else (genome.mutation_note or NO_NOTE)
    return (
        f"{index}. {genome.genome_id}, generation {genome.generation}, "
        f"fitness {fitness:.4f}: {note}"
    )
