"""Command-line interface: ``evorg run``, ``evorg estimate``, ``evorg show``, ``evorg bench``
and ``evorg serve``.

``main`` returns an exit code and never calls ``sys.exit`` itself, so tests can call it
directly. Progress goes to stderr; reports and tables go to stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Sequence
from pathlib import Path

from evorg.bench import load_benchmark, task_index
from evorg.evolution import (
    EvolutionConfig,
    GenerationRecord,
    evaluate,
    evolve,
    run_simulated,
    simulated_backends,
)
from evorg.genome import OrgGenome
from evorg.llm import SimulatedBackend, make_backend
from evorg.llm.budget import (
    BudgetedBackend,
    BudgetExceeded,
    CostBound,
    CountingBackend,
    UsageTally,
    estimate_upper_bound,
)
from evorg.mutation import Architect
from evorg.report import render_markdown

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_CAP = 3
REPORT_NAME = "report.md"
RUN_NAME = "run.json"
WEB_MISSING = 'install the web extra: pip install -e ".[web]"'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evorg", description="Evolve agent organizations and report on the results."
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    run = sub.add_parser("run", help="evolve an organization and write a report")
    run.add_argument("--backend", choices=["sim", "anthropic"], default="sim")
    _add_config_flags(run)
    run.add_argument("--out", help="output directory (default: runs/<backend>-seed<seed>)")
    run.add_argument(
        "--yes", action="store_true", help="confirm a run that calls the paid Anthropic API"
    )
    run.add_argument(
        "--max-usd",
        type=float,
        default=None,
        help="hard spending cap in USD, required with --backend anthropic",
    )

    estimate = sub.add_parser(
        "estimate", help="count the calls of a simulated run and bound the cost of a real one"
    )
    _add_config_flags(estimate)
    estimate.add_argument(
        "--large-share",
        type=float,
        default=None,
        help="fraction of calls priced on the large tier (default: the simulated run's mix)",
    )

    show = sub.add_parser("show", help="print the report saved in a run directory")
    show.add_argument("dir")

    bench = sub.add_parser("bench", help="evaluate one genome JSON file on a benchmark split")
    bench.add_argument("genome", help="path to a genome JSON file")
    bench.add_argument("--split", choices=["train", "holdout", "all"], default="train")
    bench.add_argument("--backend", choices=["sim"], default="sim")
    bench.add_argument("--seed", type=int, default=0)

    serve = sub.add_parser("serve", help="serve saved runs and the lineage explorer page")
    serve.add_argument("--runs-dir", default="runs", help="directory holding run folders")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--host", default="127.0.0.1")
    return parser


def _add_config_flags(parser: argparse.ArgumentParser) -> None:
    """The evolution settings shared by ``run`` and ``estimate``."""
    default_config = EvolutionConfig()
    parser.add_argument("--generations", type=int, default=default_config.generations)
    parser.add_argument("--population", type=int, default=default_config.population)
    parser.add_argument("--seed", type=int, default=default_config.seed)
    parser.add_argument("--cost-weight", type=float, default=default_config.cost_weight)
    parser.add_argument(
        "--eval-seeds",
        type=int,
        default=default_config.eval_seeds,
        help="simulator seeds each genome is scored on (sim backend only)",
    )


def _config_from_args(args: argparse.Namespace) -> EvolutionConfig:
    return EvolutionConfig(
        population=args.population,
        generations=args.generations,
        seed=args.seed,
        cost_weight=args.cost_weight,
        eval_seeds=args.eval_seeds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "run": _cmd_run,
        "estimate": _cmd_estimate,
        "show": _cmd_show,
        "bench": _cmd_bench,
        "serve": _cmd_serve,
    }
    return handlers[args.command](args)


def _progress(record: GenerationRecord) -> None:
    best_id, best_fitness = record.members[0]
    print(
        f"generation {record.index}: best fitness {best_fitness:.4f} ({best_id}), "
        f"{len(record.members)} members",
        file=sys.stderr,
        flush=True,
    )


def _valid_cap(max_usd: float | None) -> bool:
    return max_usd is not None and math.isfinite(max_usd) and max_usd > 0


def _cmd_run(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    out = Path(args.out) if args.out else Path("runs") / f"{args.backend}-seed{args.seed}"

    if args.backend == "anthropic" and args.eval_seeds > 1:
        print(
            "error: --eval-seeds above 1 needs the simulated backend; repeating paid "
            "Anthropic calls would multiply the cost",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if args.backend == "sim" and args.max_usd is not None:
        print("error: --max-usd applies only to --backend anthropic", file=sys.stderr)
        return EXIT_USAGE

    budget: BudgetedBackend | None = None
    try:
        if args.backend == "anthropic":
            print("warning: --backend anthropic calls the paid Anthropic API.", file=sys.stderr)
            if not args.yes:
                print("error: pass --yes to run against the paid API", file=sys.stderr)
                return EXIT_USAGE
            if not _valid_cap(args.max_usd):
                print(
                    "error: --max-usd X is required with --backend anthropic, "
                    "where X is a positive number of USD",
                    file=sys.stderr,
                )
                return EXIT_USAGE
            # One budgeted backend serves both the evaluator and the architect, so every
            # call counts against the same cap.
            budget = BudgetedBackend(make_backend("anthropic"), args.max_usd)
            architect = Architect(budget, random.Random(config.seed + 1))
            run = evolve(
                config,
                budget,
                load_benchmark("train"),
                load_benchmark("holdout"),
                architect,
                on_generation=_progress,
            )
        else:
            run = run_simulated(config, on_generation=_progress)
    except BudgetExceeded as exc:
        print(
            f"stopped: spending cap reached (${exc.spent_usd:.4f} of ${exc.limit_usd:.2f})",
            file=sys.stderr,
        )
        return EXIT_CAP
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    out.mkdir(parents=True, exist_ok=True)
    (out / RUN_NAME).write_text(json.dumps(run.to_dict(), indent=2, sort_keys=True))
    report = render_markdown(run)
    (out / REPORT_NAME).write_text(report)
    sys.stdout.write(report)
    print(f"wrote {out / RUN_NAME} and {out / REPORT_NAME}", file=sys.stderr)
    if budget is not None:
        print(
            f"spent ${budget.spent_usd:.4f} of ${budget.limit_usd:.2f} cap "
            f"across {budget.calls} calls",
            file=sys.stderr,
        )
    return EXIT_OK


def _cmd_estimate(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    if args.eval_seeds > 1:
        print(
            "error: estimate prices a real run, which uses one scoring seed; use --eval-seeds 1",
            file=sys.stderr,
        )
        return EXIT_USAGE
    try:
        bound = _counted_simulation_bound(config, args.large_share)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    _print_estimate(config, bound)
    return EXIT_OK


def _counted_simulation_bound(config: EvolutionConfig, large_share: float | None) -> CostBound:
    """Run ``config`` on the simulator with counting wrappers, then bound the real cost.

    The wrappers sit where the Anthropic backend would sit in a real run: the evaluator and
    the architect share one training backend, and holdout scoring uses the holdout backend.
    """
    tally = UsageTally()
    train, holdout = simulated_backends(config)
    counted_train = [CountingBackend(b, tally) for b in train]
    counted_holdout = [CountingBackend(b, tally) for b in holdout]
    architect = Architect(counted_train[0], random.Random(config.seed + 1))
    evolve(
        config,
        counted_train,
        load_benchmark("train"),
        load_benchmark("holdout"),
        architect,
        holdout_backends=counted_holdout,
    )
    return estimate_upper_bound(tally, large_share)


def _print_estimate(config: EvolutionConfig, bound: CostBound) -> None:
    print(
        f"evorg estimate: simulated run with {config.generations} generations, "
        f"population {config.population}, seed {config.seed}"
    )
    print()
    print(f"{'tier':<8}{'LLM calls':>12}{'upper bound (USD)':>20}")
    for tier in sorted(bound.calls):
        print(f"{tier:<8}{bound.calls[tier]:>12}{bound.usd_by_tier[tier]:>20.6f}")
    print(f"{'total':<8}{bound.total_calls:>12}{bound.total_usd:>20.6f}")
    print()
    print(
        f"Upper bound for a real run: ${bound.total_usd:.4f}. The simulated run's model "
        "choices stand in for the real run's (or the --large-share mix). The bound assumes "
        "every call uses the full output budget and 1.5x the simulated input tokens."
    )
    print(
        "Real call counts may differ, because real architect replies differ from simulated "
        "ones. The bound covers the call mix shown above."
    )


def _cmd_show(args: argparse.Namespace) -> int:
    path = Path(args.dir) / REPORT_NAME
    if not path.is_file():
        print(f"error: no report at {path}", file=sys.stderr)
        return EXIT_ERROR
    sys.stdout.write(path.read_text())
    return EXIT_OK


def _cmd_bench(args: argparse.Namespace) -> int:
    try:
        data = json.loads(Path(args.genome).read_text())
        genome = OrgGenome.from_dict(data)
        genome.validate()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: cannot load genome from {args.genome}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    tasks = load_benchmark(args.split)
    backend = SimulatedBackend(task_index(), seed=args.seed)
    ev = evaluate(genome, tasks, backend, EvolutionConfig().cost_weight)

    print(f"{'task':<26}{'passed':<8}{'tests':>8}{'attempts':>10}{'cost (USD)':>13}")
    for r in ev.results:
        print(
            f"{r.task_id:<26}{'yes' if r.passed else 'no':<8}"
            f"{r.pass_fraction * 100:>7.0f}%{r.attempts:>10}{r.cost_usd:>13.6f}"
        )
    n_passed = sum(1 for r in ev.results if r.passed)
    total_cost = sum(r.cost_usd for r in ev.results)
    print(
        f"Totals: {n_passed}/{len(ev.results)} passed ({ev.pass_rate:.1%}), "
        f"mean cost ${ev.mean_cost_usd:.6f} per task, total ${total_cost:.6f}, "
        f"fitness {ev.fitness:.4f}"
    )
    return EXIT_OK


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from evorg.web import create_app
    except ImportError:
        print(WEB_MISSING, file=sys.stderr)
        return EXIT_USAGE

    print(f"serving {args.runs_dir} at http://{args.host}:{args.port}", file=sys.stderr, flush=True)
    uvicorn.run(create_app(Path(args.runs_dir)), host=args.host, port=args.port)
    return EXIT_OK
