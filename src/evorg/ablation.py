"""Ablation: what each gene of a champion is worth.

A gene is one part of the champion that can be taken away: a role, an instruction, a model tier
above ``small``, or a retry budget above the seed's. Each variant is the champion with one gene
removed or reverted. The champion and every variant are scored on the same tasks and the same
simulator seeds, so the differences are paired. Pass rate and cost are paired by (seed, task);
fitness is paired by seed. Every interval is a paired bootstrap at 95%. Every p-value is a
paired sign-flip permutation test on the same pairs, Holm-adjusted across the rows of one
ablation, and a verdict calls a change real only when its adjusted p is below ``ALPHA``.

Nothing here touches the disk or calls a paid API. ``ablate`` is deterministic given its inputs.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, replace

from evorg.bench import Task, task_index
from evorg.evolution import HOLDOUT_SEED_OFFSET, TRAIN_SEED_STEP, Evaluation, evaluate
from evorg.genome import GenomeError, OrgGenome
from evorg.llm.simulated import SimulatedBackend
from evorg.report import SIM_NOTE
from evorg.stats import holm_adjust, paired_bootstrap_ci, paired_permutation_p

CI_LEVEL = 0.95
CI_BOOTSTRAP_SEED = 0  # fixed, so the same runs always print the same intervals
ALPHA = 0.05  # a change is called real only when its Holm-adjusted p is below this
PERMUTATION_SEED = 0  # fixed, so the same runs always print the same p-values
PERMUTATION_REPS = 2000  # used only when a family has more pairs than the exact limit
SMALL_TIER = "small"


@dataclass(frozen=True)
class Variant:
    """The champion with one gene removed or reverted."""

    name: str  # stable id such as "role:reviewer" or "instruction:coder[1]"
    kind: str  # "role", "instruction", "model" or "retries"
    label: str  # plain English, e.g. "removing the reviewer"
    genome: OrgGenome


@dataclass(frozen=True)
class Skipped:
    """A gene whose removal would give an invalid genome, with the reason."""

    name: str
    label: str
    reason: str


@dataclass(frozen=True)
class Delta:
    """Variant minus champion: the mean over pairs and its paired 95% interval."""

    mean: float
    low: float
    high: float


@dataclass(frozen=True)
class AblationRow:
    variant: Variant
    pass_rate: Delta  # fraction of tasks passed, so 0.1 is ten points
    cost: Delta  # USD per task
    fitness: Delta
    verdict: str
    pass_p: float = 1.0  # Holm-adjusted sign-flip p, pass rate paired by (seed, task)
    cost_p: float = 1.0  # Holm-adjusted sign-flip p, cost paired by (seed, task)
    fitness_p: float = 1.0  # Holm-adjusted sign-flip p, fitness paired by seed


@dataclass(frozen=True)
class AblationResult:
    champion_id: str
    split: str
    seeds: tuple[int, ...]
    n_tasks: int
    champion_pass_rate: float  # mean over seeds
    champion_cost: float  # mean over seeds, USD per task
    champion_fitness: float  # mean over seeds
    rows: tuple[AblationRow, ...]  # most harmful removal first
    skipped: tuple[Skipped, ...]


def build_variants(champion: OrgGenome, seed: OrgGenome) -> tuple[list[Variant], list[Skipped]]:
    """Every single-gene variant of ``champion``, in a fixed order, plus the invalid ones.

    Genes are visited role by role: the role itself, then each of its instructions, then its
    model tier if it is above small. The retry gene exists only when the champion's budget
    differs from the seed's. A variant that fails ``OrgGenome.validate`` is skipped and listed
    with the validation message.
    """
    variants: list[Variant] = []
    skipped: list[Skipped] = []

    def add(name: str, kind: str, label: str, roles: Sequence, retries: int) -> None:
        child = replace(
            champion,
            roles=tuple(roles),
            max_retries=retries,
            parent_ids=(champion.genome_id,),
            mutation_note=f"ablation: {label}",
        )
        try:
            child.validate()
        except GenomeError as exc:
            skipped.append(Skipped(name=name, label=label, reason=str(exc)))
            return
        variants.append(Variant(name=name, kind=kind, label=label, genome=child))

    roles = list(champion.roles)
    for i, role in enumerate(roles):
        if role.kind == "coder":
            skipped.append(
                Skipped(
                    name=f"role:{role.name}",
                    label=f"removing the {role.name}",
                    reason="the coder cannot be removed; an organization needs exactly one",
                )
            )
        else:
            rest = roles[:i] + roles[i + 1 :]
            add(
                f"role:{role.name}", "role", f"removing the {role.name}", rest, champion.max_retries
            )

        for j, text in enumerate(role.instructions):
            kept = replace(role, instructions=role.instructions[:j] + role.instructions[j + 1 :])
            label = f"removing instruction '{text}' from {role.name}"
            add(
                f"instruction:{role.name}[{j}]",
                "instruction",
                label,
                roles[:i] + [kept] + roles[i + 1 :],
                champion.max_retries,
            )

        if role.model != SMALL_TIER:
            reverted = replace(role, model=SMALL_TIER)
            add(
                f"model:{role.name}",
                "model",
                f"switching the {role.name} back to the {SMALL_TIER} model",
                roles[:i] + [reverted] + roles[i + 1 :],
                champion.max_retries,
            )

    if champion.max_retries != seed.max_retries:
        add(
            "retries",
            "retries",
            f"setting retries back to {seed.max_retries} (from {champion.max_retries})",
            roles,
            seed.max_retries,
        )
    return variants, skipped


def seed_offsets(base_seed: int, split: str, count: int) -> list[int]:
    """Simulator seeds for ``count`` scoring runs, matching the evolution run's convention.

    The holdout split uses the holdout offset that ``simulated_backends`` uses, so these seeds
    never coincide with training seeds.
    """
    offset = HOLDOUT_SEED_OFFSET if split == "holdout" else 0
    return [base_seed + offset + TRAIN_SEED_STEP * k for k in range(count)]


def champion_and_seed(data: dict) -> tuple[OrgGenome, OrgGenome, float]:
    """The champion genome, the seed genome and the cost weight, from a saved run.json dict."""
    if data.get("backend") != "sim":
        raise ValueError(
            f"ablation uses the simulated backend; this run used {data.get('backend')!r}"
        )
    genomes = data["genomes"]
    champion = OrgGenome.from_dict(genomes[data["champion_id"]])
    seed = OrgGenome.from_dict(genomes[data["seed_id"]])
    return champion, seed, float(data["config"]["cost_weight"])


def complexity_weight_of(data: dict) -> float:
    """The parsimony weight from a saved run.json dict; 0 when the run did not record one."""
    return float(data["config"].get("complexity_weight", 0.0))


def _interval(a: Sequence[float], b: Sequence[float]) -> Delta:
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    low, high = paired_bootstrap_ci(a, b, level=CI_LEVEL, seed=CI_BOOTSTRAP_SEED)
    return Delta(mean=statistics.fmean(diffs), low=low, high=high)


def _raw_p(a: Sequence[float], b: Sequence[float]) -> float:
    """Unadjusted two-sided sign-flip p-value; constant or identical arms give 1."""
    return paired_permutation_p(a, b, reps=PERMUTATION_REPS, seed=PERMUTATION_SEED)


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _pass_values(ev: Sequence[Evaluation]) -> list[float]:
    return [1.0 if r.passed else 0.0 for e in ev for r in e.results]


def _cost_values(ev: Sequence[Evaluation]) -> list[float]:
    return [r.cost_usd for e in ev for r in e.results]


def _fitness_values(ev: Sequence[Evaluation]) -> list[float]:
    return [e.fitness for e in ev]


def _verdict(
    label: str,
    pass_rate: Delta,
    cost: Delta,
    fitness: Delta,
    champ_cost: float,
    *,
    pass_p: float = 1.0,
    cost_p: float = 1.0,
    fitness_p: float = 1.0,
) -> str:
    """One sentence per metric. A change is called real only when its adjusted p is below ALPHA.

    The p-values are Holm-adjusted across the rows of the ablation, so they already allow for
    testing many genes at once. Anything else is reported as not clear from these seeds.
    """
    points = pass_rate.mean * 100
    if round(abs(points)) == 0:
        pass_clause = "leaves pass rate about unchanged"
    else:
        detail = (
            f"(95% CI {pass_rate.low * 100:.0f} to {pass_rate.high * 100:.0f}, "
            f"adjusted p {_fmt_p(pass_p)})"
        )
        if pass_p < ALPHA:
            verb = "costs" if points < 0 else "adds"
            pass_clause = f"{verb} {abs(points):.0f} points of pass rate {detail}"
        else:
            pass_clause = (
                f"moves pass rate by {abs(points):.0f} points, not clear from these seeds {detail}"
            )

    if champ_cost > 0:
        pct = cost.mean / champ_cost * 100
        if round(abs(pct)) == 0:
            cost_clause = "leaves cost about unchanged"
        elif cost_p < ALPHA:
            verb = "saves" if pct < 0 else "adds"
            cost_clause = f"{verb} {abs(pct):.0f}% cost (adjusted p {_fmt_p(cost_p)})"
        else:
            cost_clause = f"changes cost by {pct:+.0f}%, not clear from these seeds"
    else:
        cost_clause = "cost change is not clear from these seeds"

    if fitness_p < ALPHA and fitness.mean < 0:
        tail = "Keeping it is worth its cost: fitness drops without it."
    elif fitness_p < ALPHA and fitness.mean > 0:
        tail = "Fitness rises without it, so it does not earn its cost."
    else:
        tail = "The fitness change is not clear from these seeds."
    sentence = f"{label[0].upper()}{label[1:]} {pass_clause} and {cost_clause}."
    return f"{sentence} {tail}"


def ablate(
    champion: OrgGenome,
    seed: OrgGenome,
    tasks: Sequence[Task],
    seeds: Sequence[int],
    cost_weight: float,
    split: str = "custom",
    complexity_weight: float = 0.0,
) -> AblationResult:
    """Score ``champion`` and each of its single-gene variants on ``tasks`` for each seed.

    Each variant's deltas are taken against the champion on the same (seed, task) pairs. Rows
    come back sorted by fitness change, most harmful removal first, ties broken by name.
    ``complexity_weight`` is the run's parsimony weight, so fitness matches the run's own.
    """
    if not seeds:
        raise ValueError("ablate needs at least one simulator seed")
    if not tasks:
        raise ValueError("ablate needs at least one task")
    index = task_index()
    variants, skipped = build_variants(champion, seed)

    def score(genome: OrgGenome) -> list[Evaluation]:
        return [
            evaluate(
                genome,
                tasks,
                SimulatedBackend(index, seed=s),
                cost_weight,
                complexity_weight=complexity_weight,
            )
            for s in seeds
        ]

    champ_evs = score(champion)
    champ_pass = _pass_values(champ_evs)
    champ_costs = _cost_values(champ_evs)
    champ_fitness = _fitness_values(champ_evs)
    champ_cost = statistics.fmean(champ_costs)

    measured = []
    for variant in variants:
        evs = score(variant.genome)
        measured.append(
            (
                variant,
                _interval(_pass_values(evs), champ_pass),
                _interval(_cost_values(evs), champ_costs),
                _interval(_fitness_values(evs), champ_fitness),
                _raw_p(_pass_values(evs), champ_pass),
                _raw_p(_cost_values(evs), champ_costs),
                _raw_p(_fitness_values(evs), champ_fitness),
            )
        )
    # One Holm family per metric, across the rows of this ablation.
    pass_adj = holm_adjust([m[4] for m in measured])
    cost_adj = holm_adjust([m[5] for m in measured])
    fitness_adj = holm_adjust([m[6] for m in measured])

    rows: list[AblationRow] = []
    for i, (variant, pass_rate, cost, fitness, _, _, _) in enumerate(measured):
        verdict = _verdict(
            variant.label,
            pass_rate,
            cost,
            fitness,
            champ_cost,
            pass_p=pass_adj[i],
            cost_p=cost_adj[i],
            fitness_p=fitness_adj[i],
        )
        rows.append(
            AblationRow(
                variant,
                pass_rate,
                cost,
                fitness,
                verdict,
                pass_p=pass_adj[i],
                cost_p=cost_adj[i],
                fitness_p=fitness_adj[i],
            )
        )
    rows.sort(key=lambda r: (r.fitness.mean, r.variant.name))

    return AblationResult(
        champion_id=champion.genome_id,
        split=split,
        seeds=tuple(seeds),
        n_tasks=len(tasks),
        champion_pass_rate=statistics.fmean(e.pass_rate for e in champ_evs),
        champion_cost=statistics.fmean(e.mean_cost_usd for e in champ_evs),
        champion_fitness=statistics.fmean(e.fitness for e in champ_evs),
        rows=tuple(rows),
        skipped=tuple(skipped),
    )


def _signed_pts(delta: Delta) -> str:
    return f"{delta.mean * 100:+.1f} pts ({delta.low * 100:+.1f} to {delta.high * 100:+.1f})"


def _signed_usd(delta: Delta) -> str:
    return f"{delta.mean:+.6f} ({delta.low:+.6f} to {delta.high:+.6f})"


def _signed_fitness(delta: Delta) -> str:
    return f"{delta.mean:+.4f} ({delta.low:+.4f} to {delta.high:+.4f})"


def render_ablation(result: AblationResult) -> str:
    """Render an ablation as markdown. Makes no calls and reads nothing but ``result``."""
    seeds = ", ".join(str(s) for s in result.seeds)
    lines: list[str] = [
        "# evorg ablation",
        "",
        (
            f"Champion {result.champion_id} on the {result.split} split: "
            f"{result.n_tasks} tasks, simulator seeds {seeds}. Champion pass rate "
            f"{result.champion_pass_rate:.1%}, mean cost ${result.champion_cost:.6f} per task, "
            f"fitness {result.champion_fitness:.4f}."
        ),
        "",
        (
            "Each row removes or reverts one gene from the champion. Changes are variant minus "
            "champion. Pass rate and cost are paired by (seed, task); fitness is paired by seed. "
            "Brackets are 95% paired bootstrap intervals. Rows are sorted by fitness change, "
            "most harmful removal first."
        ),
        "",
        "## Gene values",
        "",
        (
            "| gene | pass rate change | cost change per task (USD) | fitness change | "
            "adjusted p (fitness) | verdict |"
        ),
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in result.rows:
        lines.append(
            f"| {row.variant.name} | {_signed_pts(row.pass_rate)} | "
            f"{_signed_usd(row.cost)} | {_signed_fitness(row.fitness)} | "
            f"{_fmt_p(row.fitness_p)} | {row.verdict} |"
        )
    if not result.rows:
        lines.append("| (none) | | | | | the champion has no genes to ablate |")
    lines += [
        "",
        (
            "Adjusted p is a two-sided sign-flip permutation p-value on the paired differences, "
            "Holm-Bonferroni adjusted across the rows of this table so that testing many genes at "
            "once does not inflate false alarms, and a change is called real only when it is below "
            "0.05; otherwise the verdict says the change is not clear from these seeds."
        ),
    ]

    lines += ["", "## Skipped variants", ""]
    if result.skipped:
        for item in result.skipped:
            lines.append(f"- {item.name}: {item.label}, skipped because {item.reason}")
    else:
        lines.append("None. Every gene's removal gives a valid genome.")
    lines += ["", SIM_NOTE, ""]
    return "\n".join(lines)
