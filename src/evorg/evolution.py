"""Evolve organization genomes: evaluate, select, mutate, and report on a held-out split.

The loop follows design section 4.7. Generation 0 is the seed plus architect proposals for it.
Each later generation makes one child per parent slot (tournament selection, then one
architect-proposed mutation), evaluates the children, and keeps the elite plus tournament
winners from the pool of parents and children. Evaluations are cached by genome id, and every
choice that could vary is ordered by a ranking key, so a run depends only on its config.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from evorg import pareto
from evorg.bench import Task, load_benchmark, task_index
from evorg.crossover import crossover
from evorg.executor import TaskResult, run_task
from evorg.genome import OrgGenome, seed_genome
from evorg.llm.base import LLMBackend
from evorg.llm.simulated import SimulatedBackend
from evorg.mutation import Architect, MutationError, apply_mutation, random_mutation

REPORT_CHAR_LIMIT = 1500
MAX_TOP_UP_TRIES = 1000
FLOAT_DIGITS = 6
TRAIN_SEED_STEP = 1000  # spacing between the simulator seeds of one run
HOLDOUT_SEED_OFFSET = 500_000  # holdout seeds sit far above every training seed
CROSSOVER_PAIR_TRIES = 3  # draws for a second parent that differs from the first
CROSSOVER_MUTATE_PROB = 0.5  # chance that a crossover child is also mutated
SELECTION_MODES = ("scalar", "pareto")


@dataclass(frozen=True)
class EvolutionConfig:
    population: int = 6
    generations: int = 5
    tournament_size: int = 3
    elite: int = 1
    cost_weight: float = 10.0  # fitness = pass_rate - cost_weight * mean cost per task
    seed: int = 0
    eval_seeds: int = 1  # simulator seeds each genome is scored on, averaged
    workers: int = 1  # tasks graded or called at once; never changes the results
    crossover_rate: float = 0.0  # chance that a child is bred by crossover, not mutation
    selection: str = "scalar"  # "scalar": rank by fitness; "pareto": rank by front, then crowding


@dataclass(frozen=True)
class Evaluation:
    genome_id: str
    pass_rate: float
    mean_cost_usd: float
    fitness: float
    results: tuple[TaskResult, ...]

    def report(self) -> str:
        """A short summary for the architect: the pass rate, then one line per failed task."""
        n_tasks = len(self.results)
        n_passed = sum(1 for r in self.results if r.passed)
        header = (
            f"Pass rate {n_passed}/{n_tasks} ({self.pass_rate * 100:.0f}%), "
            f"mean cost ${self.mean_cost_usd:.5f} per task."
        )
        failed = [r for r in self.results if not r.passed]
        lines = [header]
        for i, result in enumerate(failed):
            line = (
                f"- {result.task_id}: {result.pass_fraction * 100:.0f}% of tests passed, "
                f"{result.attempts} {'attempt' if result.attempts == 1 else 'attempts'}"
            )
            remaining = len(failed) - i - 1
            candidate = "\n".join([*lines, line])
            # Reserve room for a trailing "... N more" line when the report would overflow.
            tail = f"\n- ... {remaining} more" if remaining else ""
            if len(candidate) + len(tail) > REPORT_CHAR_LIMIT:
                lines.append(f"- ... {len(failed) - i} more")
                break
            lines.append(line)
        return "\n".join(lines)


@dataclass
class GenerationRecord:
    index: int
    members: list[tuple[str, float]]  # (genome_id, fitness), best first
    best_id: str


@dataclass
class RunResult:
    config: EvolutionConfig
    backend: str
    generations: list[GenerationRecord]
    genomes: dict[str, OrgGenome]  # every genome ever evaluated
    evaluations: dict[str, Evaluation]  # train evaluations
    champion_id: str
    seed_id: str
    holdout: dict[str, Evaluation]  # champion and seed on held-out tasks
    architect_stats: dict = field(default_factory=dict)
    crossovers: int = 0  # children bred by crossover
    frontier: list[str] = field(default_factory=list)  # front 0 over train evals, cost ascending

    def lineage(self, genome_id: str) -> list[OrgGenome]:
        """The chain from a root genome (no parents) down to ``genome_id``, root first."""
        chain = [self.genomes[genome_id]]
        while chain[-1].parent_ids:
            parent_id = chain[-1].parent_ids[0]
            if parent_id not in self.genomes:
                raise KeyError(f"parent {parent_id} of {chain[-1].genome_id} was not recorded")
            chain.append(self.genomes[parent_id])
        chain.reverse()
        return chain

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": _config_dict(self.config),
            "backend": self.backend,
            "generations": [
                {
                    "index": record.index,
                    "members": [[gid, _r(fitness)] for gid, fitness in record.members],
                    "best_id": record.best_id,
                }
                for record in self.generations
            ],
            "genomes": {gid: genome.to_dict() for gid, genome in self.genomes.items()},
            "evaluations": {gid: _evaluation_dict(ev) for gid, ev in self.evaluations.items()},
            "champion_id": self.champion_id,
            "seed_id": self.seed_id,
            "holdout": {name: _evaluation_dict(ev) for name, ev in self.holdout.items()},
            "crossovers": self.crossovers,
            "frontier": list(self.frontier),
            "architect_stats": {
                key: _r(value) if isinstance(value, float) else value
                for key, value in self.architect_stats.items()
            },
        }


def _r(value: float) -> float:
    return round(float(value), FLOAT_DIGITS)


def _config_dict(config: EvolutionConfig) -> dict[str, Any]:
    """The config as saved in run.json. ``workers`` is left out: it changes only the speed."""
    data = asdict(config)
    data.pop("workers")
    return data


def _evaluation_dict(ev: Evaluation) -> dict[str, Any]:
    return {
        "pass_rate": _r(ev.pass_rate),
        "mean_cost_usd": _r(ev.mean_cost_usd),
        "fitness": _r(ev.fitness),
        "per_task": [
            {
                "task_id": r.task_id,
                "passed": r.passed,
                "pass_fraction": _r(r.pass_fraction),
                "attempts": r.attempts,
                "calls": r.calls,
                "cost_usd": _r(r.cost_usd),
            }
            for r in ev.results
        ],
    }


def _check_workers(workers: int) -> None:
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")


def _summarise(
    genome: OrgGenome, results: tuple[TaskResult, ...], cost_weight: float
) -> Evaluation:
    n = len(results)
    pass_rate = sum(1 for r in results if r.passed) / n if n else 0.0
    mean_cost = sum(r.cost_usd for r in results) / n if n else 0.0
    return Evaluation(
        genome_id=genome.genome_id,
        pass_rate=pass_rate,
        mean_cost_usd=mean_cost,
        fitness=pass_rate - cost_weight * mean_cost,
        results=results,
    )


def evaluate(
    genome: OrgGenome,
    tasks: Sequence[Task],
    backend: LLMBackend,
    cost_weight: float,
    workers: int = 1,
) -> Evaluation:
    """Run ``genome`` on every task and score it.

    ``pass_rate`` is the share of tasks passed, ``mean_cost_usd`` the mean cost per task, and
    ``fitness = pass_rate - cost_weight * mean_cost_usd``. Nothing is rounded. With
    ``workers > 1`` up to that many tasks run on a thread pool; ``results`` stay in task order,
    so the Evaluation is the same as with one worker, as long as the backend is thread-safe.
    """
    _check_workers(workers)
    if workers == 1:
        results = tuple(run_task(genome, task, backend) for task in tasks)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = tuple(pool.map(lambda task: run_task(genome, task, backend), tasks))
    return _summarise(genome, results, cost_weight)


def _run_grid(
    genome: OrgGenome,
    tasks: Sequence[Task],
    backends: Sequence[LLMBackend],
    workers: int,
) -> list[tuple[TaskResult, ...]]:
    """Run ``genome`` on every (backend, task) pair on one pool; one result tuple per backend."""
    pairs = [(backend, task) for backend in backends for task in tasks]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        flat = list(pool.map(lambda pair: run_task(genome, pair[1], pair[0]), pairs))
    n = len(tasks)
    return [tuple(flat[i * n : (i + 1) * n]) for i in range(len(backends))]


def evaluate_many(
    genome: OrgGenome,
    tasks: Sequence[Task],
    backends: Sequence[LLMBackend],
    cost_weight: float,
    workers: int = 1,
) -> Evaluation:
    """Score ``genome`` on ``tasks`` once per backend and average the scores.

    ``pass_rate``, ``mean_cost_usd`` and ``fitness`` are the means over the backends, and
    ``results`` holds every backend's per-task results in backend order. With one backend this
    is exactly ``evaluate``. ``workers`` runs the whole backend x task grid on one thread pool.
    """
    if not backends:
        raise ValueError("evaluate_many needs at least one backend")
    _check_workers(workers)
    if len(backends) == 1 and workers == 1:
        return evaluate(genome, tasks, backends[0], cost_weight)
    if workers == 1:
        evals = [evaluate(genome, tasks, backend, cost_weight) for backend in backends]
    else:
        grid = _run_grid(genome, tasks, backends, workers)
        evals = [_summarise(genome, results, cost_weight) for results in grid]
    n = len(evals)
    return Evaluation(
        genome_id=genome.genome_id,
        pass_rate=sum(ev.pass_rate for ev in evals) / n,
        mean_cost_usd=sum(ev.mean_cost_usd for ev in evals) / n,
        fitness=sum(ev.fitness for ev in evals) / n,
        results=tuple(result for ev in evals for result in ev.results),
    )


def _rank_key(ev: Evaluation) -> tuple[float, float, str]:
    """Sort key, best first: higher fitness, then lower mean cost, then genome id."""
    return (-ev.fitness, ev.mean_cost_usd, ev.genome_id)


def selection_keys(
    members: Sequence[OrgGenome], evaluations: dict[str, Evaluation], selection: str
) -> dict[str, tuple]:
    """Sort key per genome id for ``members``; sorting ascending by key puts the best first.

    ``"scalar"`` uses ``_rank_key``. ``"pareto"`` ranks the whole group at once by Pareto front
    and crowding distance (see ``pareto.pareto_rank``), so keys are only comparable within one
    call.
    """
    ids = [g.genome_id for g in members]
    if selection == "pareto":
        points = pareto.points_from_evaluations({gid: evaluations[gid] for gid in ids})
        ranks = pareto.pareto_rank(points)
        return {gid: pareto.pareto_sort_key(ranks[gid], gid) for gid in ids}
    return {gid: _rank_key(evaluations[gid]) for gid in ids}


def _tournament(
    pool: Sequence[OrgGenome],
    keys: dict[str, tuple],
    size: int,
    rng: random.Random,
) -> OrgGenome:
    """The best of ``size`` random entrants from ``pool``, judged by precomputed ``keys``."""
    entrants = rng.sample(list(pool), min(size, len(pool)))
    return min(entrants, key=lambda g: keys[g.genome_id])


def _validate_config(config: EvolutionConfig) -> None:
    if config.population < 1:
        raise ValueError(f"population must be at least 1, got {config.population}")
    if config.generations < 0:
        raise ValueError(f"generations must be at least 0, got {config.generations}")
    if config.tournament_size < 1:
        raise ValueError(f"tournament_size must be at least 1, got {config.tournament_size}")
    if not 0 <= config.elite <= config.population:
        raise ValueError(f"elite must be between 0 and population, got {config.elite}")
    if config.eval_seeds < 1:
        raise ValueError(f"eval_seeds must be at least 1, got {config.eval_seeds}")
    _check_workers(config.workers)
    if not 0 <= config.crossover_rate <= 1:
        raise ValueError(f"crossover_rate must be between 0 and 1, got {config.crossover_rate}")
    if config.selection not in SELECTION_MODES:
        raise ValueError(
            f"selection must be 'scalar' or 'pareto', got {config.selection!r}"
        )


def _as_backends(backend: LLMBackend | Sequence[LLMBackend]) -> list[LLMBackend]:
    if isinstance(backend, Sequence):
        return list(backend)
    return [backend]


def evolve(
    config: EvolutionConfig,
    backend: LLMBackend | Sequence[LLMBackend],
    train: Sequence[Task],
    holdout: Sequence[Task],
    architect: Architect,
    on_generation: Callable[[GenerationRecord], None] | None = None,
    holdout_backends: Sequence[LLMBackend] | None = None,
) -> RunResult:
    """Run the evolutionary search described in design section 4.7.

    ``backend`` scores genomes on the training tasks; each genome's score is the mean over all
    of them. ``holdout_backends`` scores the final champion and seed on the held-out tasks and
    defaults to ``backend``. The architect keeps the backend it was built with.
    """
    _validate_config(config)
    rng = random.Random(config.seed)
    train_backends = _as_backends(backend)
    eval_backends = train_backends if holdout_backends is None else list(holdout_backends)

    genomes: dict[str, OrgGenome] = {}
    evaluations: dict[str, Evaluation] = {}
    generations: list[GenerationRecord] = []

    def register(genome: OrgGenome) -> OrgGenome:
        return genomes.setdefault(genome.genome_id, genome)

    def ensure_evaluated(genome: OrgGenome) -> None:
        if genome.genome_id not in evaluations:
            evaluations[genome.genome_id] = evaluate_many(
                genome, train, train_backends, config.cost_weight, workers=config.workers
            )

    def sorted_population(members: Sequence[OrgGenome]) -> list[OrgGenome]:
        keys = selection_keys(members, evaluations, config.selection)
        return sorted(members, key=lambda g: keys[g.genome_id])

    def record(index: int, members: Sequence[OrgGenome]) -> None:
        # Records report progress by fitness in both modes, so the best-fitness line stays
        # comparable; selection order only drives tournaments and survivors.
        ordered = sorted(members, key=lambda g: _rank_key(evaluations[g.genome_id]))
        generation = GenerationRecord(
            index=index,
            members=[(g.genome_id, evaluations[g.genome_id].fitness) for g in ordered],
            best_id=ordered[0].genome_id,
        )
        generations.append(generation)
        if on_generation is not None:
            on_generation(generation)

    # Generation 0: the seed plus single mutations of it, without duplicate ids.
    seed = register(seed_genome())
    ensure_evaluated(seed)
    population: list[OrgGenome] = [seed]
    ids = {seed.genome_id}
    need = config.population - 1
    if need > 0:
        for m in architect.propose(seed, "", n=need):
            if len(population) >= config.population:
                break
            child = _try_apply(seed, m)
            if child is None or child.genome_id in ids:
                continue
            population.append(register(child))
            ids.add(child.genome_id)
    tries = 0
    while len(population) < config.population:
        tries += 1
        if tries > MAX_TOP_UP_TRIES:
            raise RuntimeError("could not build a population of distinct genomes")
        child = _try_apply(seed, random_mutation(seed, rng))
        if child is None or child.genome_id in ids:
            continue
        population.append(register(child))
        ids.add(child.genome_id)
    for genome in population:
        ensure_evaluated(genome)
    population = sorted_population(population)
    record(0, population)

    crossovers = 0
    for gen in range(1, config.generations + 1):
        children: list[OrgGenome] = []
        distinct = len({g.genome_id for g in population}) >= 2
        keys = selection_keys(population, evaluations, config.selection)
        for _ in range(config.population):
            child, crossed = _make_child(
                population, keys, evaluations, config, architect, rng, distinct
            )
            crossovers += crossed
            child = register(child)
            ensure_evaluated(child)
            children.append(child)

        pool: dict[str, OrgGenome] = {}
        for genome in [*population, *children]:
            pool.setdefault(genome.genome_id, genome)
        ranked = sorted_population(list(pool.values()))

        next_population = ranked[: config.elite]
        remaining = ranked[config.elite :]
        remaining_keys = selection_keys(remaining, evaluations, config.selection)
        while len(next_population) < config.population and remaining:
            winner = _tournament(remaining, remaining_keys, config.tournament_size, rng)
            remaining.remove(winner)
            next_population.append(winner)
        population = sorted_population(next_population)
        record(gen, population)

    champion_id = min(evaluations.values(), key=_rank_key).genome_id
    seed_id = seed.genome_id
    holdout_evals = {
        "champion": evaluate_many(
            genomes[champion_id],
            holdout,
            eval_backends,
            config.cost_weight,
            workers=config.workers,
        ),
        "seed": evaluate_many(
            seed, holdout, eval_backends, config.cost_weight, workers=config.workers
        ),
    }
    front = pareto.frontier(pareto.points_from_evaluations(evaluations))
    return RunResult(
        config=config,
        backend=train_backends[0].name,
        generations=generations,
        genomes=genomes,
        evaluations=evaluations,
        champion_id=champion_id,
        seed_id=seed_id,
        holdout=holdout_evals,
        architect_stats=dict(architect.stats),
        crossovers=crossovers,
        frontier=[point.key for point in front],
    )


def _try_apply(genome: OrgGenome, m: Any) -> OrgGenome | None:
    try:
        return apply_mutation(genome, m)
    except MutationError:
        return None


def _make_child(
    population: Sequence[OrgGenome],
    keys: dict[str, tuple],
    evaluations: dict[str, Evaluation],
    config: EvolutionConfig,
    architect: Architect,
    rng: random.Random,
    distinct: bool,
) -> tuple[OrgGenome, int]:
    """One child for the next generation, and 1 if it was bred by crossover, else 0.

    With probability ``config.crossover_rate`` (and two distinct genomes to choose from) two
    tournament winners are crossed. Otherwise the child is one mutation of a tournament winner.
    ``keys`` are the selection keys of ``population``. The rng is not touched at all when the
    rate is zero.
    """
    if distinct and config.crossover_rate > 0 and rng.random() < config.crossover_rate:
        pair = _crossover_pair(population, keys, config.tournament_size, rng)
        if pair is not None:
            return _crossover_child(pair, evaluations, architect, rng), 1
    parent = _tournament(population, keys, config.tournament_size, rng)
    return _mutate(parent, evaluations, architect, rng), 0


def _crossover_pair(
    population: Sequence[OrgGenome],
    keys: dict[str, tuple],
    tournament_size: int,
    rng: random.Random,
) -> tuple[OrgGenome, OrgGenome] | None:
    """Two tournament winners with different ids, or None if the second draws keep matching."""
    first = _tournament(population, keys, tournament_size, rng)
    for _ in range(CROSSOVER_PAIR_TRIES):
        second = _tournament(population, keys, tournament_size, rng)
        if second.genome_id != first.genome_id:
            return first, second
    return None


def _crossover_child(
    pair: tuple[OrgGenome, OrgGenome],
    evaluations: dict[str, Evaluation],
    architect: Architect,
    rng: random.Random,
) -> OrgGenome:
    """Cross two parents, then mutate the child with probability ``CROSSOVER_MUTATE_PROB``.

    The child is unevaluated, so the architect is shown the first parent's failure report.
    Lineage keeps both parents, the crossover generation and a note that reads
    ``crossover of A x B; <mutation note>``.
    """
    first, second = pair
    child = crossover(first, second, rng)
    if rng.random() >= CROSSOVER_MUTATE_PROB:
        return child
    mutated = _mutated(child, evaluations[first.genome_id].report(), architect, rng)
    return replace(
        mutated,
        parent_ids=child.parent_ids,
        generation=child.generation,
        mutation_note=f"{child.mutation_note}; {mutated.mutation_note}",
    )


def _mutate(
    parent: OrgGenome,
    evaluations: dict[str, Evaluation],
    architect: Architect,
    rng: random.Random,
) -> OrgGenome:
    """One child of ``parent``: the architect's proposal, or a random mutation if it fails."""
    return _mutated(parent, evaluations[parent.genome_id].report(), architect, rng)


def _mutated(
    genome: OrgGenome, report: str, architect: Architect, rng: random.Random
) -> OrgGenome:
    """``genome`` with the architect's proposal applied, or a random mutation if that fails."""
    proposal = architect.propose(genome, report, n=1)[0]
    try:
        return apply_mutation(genome, proposal)
    except MutationError:
        return apply_mutation(genome, random_mutation(genome, rng))


def simulated_backends(
    config: EvolutionConfig,
) -> tuple[list[SimulatedBackend], list[SimulatedBackend]]:
    """Training and holdout simulators for ``config``, with disjoint seeds.

    Training seeds are ``config.seed + 1000 * k``, so seed 0 is the backend a single-seed run
    always used. Holdout seeds start ``500_000`` above the seed and never coincide with a
    training seed, so held-out scores come from simulator runs selection never saw.
    """
    index = task_index()
    train = [
        SimulatedBackend(index, seed=config.seed + TRAIN_SEED_STEP * k)
        for k in range(config.eval_seeds)
    ]
    holdout = [
        SimulatedBackend(index, seed=config.seed + HOLDOUT_SEED_OFFSET + TRAIN_SEED_STEP * k)
        for k in range(max(1, config.eval_seeds))
    ]
    return train, holdout


def run_simulated(
    config: EvolutionConfig,
    on_generation: Callable[[GenerationRecord], None] | None = None,
    wrap: Callable[[LLMBackend, str], LLMBackend] | None = None,
) -> RunResult:
    """Evolve on the simulated backends, using the built-in train and holdout splits.

    ``wrap(backend, stream)`` may replace each simulator backend, for example with a recording
    or replay backend. The stream label is unique per simulator seed, because each seed
    answers the same request differently.
    """
    _validate_config(config)
    sim_train, sim_holdout = simulated_backends(config)
    train_backends: Sequence[LLMBackend] = sim_train
    holdout_backends: Sequence[LLMBackend] = sim_holdout
    if wrap is not None:
        train_backends = [wrap(b, f"train:{b.seed}") for b in sim_train]
        holdout_backends = [wrap(b, f"holdout:{b.seed}") for b in sim_holdout]
    architect = Architect(train_backends[0], random.Random(config.seed + 1))
    return evolve(
        config,
        train_backends,
        load_benchmark("train"),
        load_benchmark("holdout"),
        architect,
        on_generation=on_generation,
        holdout_backends=holdout_backends,
    )
