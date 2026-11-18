"""Command-line interface: ``evorg run``, ``evorg show`` and ``evorg bench``.

``main`` returns an exit code and never calls ``sys.exit`` itself, so tests can call it
directly. Progress goes to stderr; reports and tables go to stdout.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path

from evorg.bench import load_benchmark, task_index
from evorg.evolution import EvolutionConfig, GenerationRecord, evaluate, evolve, run_simulated
from evorg.genome import OrgGenome
from evorg.llm import SimulatedBackend, make_backend
from evorg.mutation import Architect
from evorg.report import render_markdown

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
REPORT_NAME = "report.md"
RUN_NAME = "run.json"


def build_parser() -> argparse.ArgumentParser:
    default_config = EvolutionConfig()
    parser = argparse.ArgumentParser(
        prog="evorg", description="Evolve agent organizations and report on the results."
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    run = sub.add_parser("run", help="evolve an organization and write a report")
    run.add_argument("--backend", choices=["sim", "anthropic"], default="sim")
    run.add_argument("--generations", type=int, default=default_config.generations)
    run.add_argument("--population", type=int, default=default_config.population)
    run.add_argument("--seed", type=int, default=default_config.seed)
    run.add_argument("--cost-weight", type=float, default=default_config.cost_weight)
    run.add_argument(
        "--eval-seeds",
        type=int,
        default=default_config.eval_seeds,
        help="simulator seeds each genome is scored on (sim backend only)",
    )
    run.add_argument("--out", help="output directory (default: runs/<backend>-seed<seed>)")
    run.add_argument(
        "--yes", action="store_true", help="confirm a run that calls the paid Anthropic API"
    )

    show = sub.add_parser("show", help="print the report saved in a run directory")
    show.add_argument("dir")

    bench = sub.add_parser("bench", help="evaluate one genome JSON file on a benchmark split")
    bench.add_argument("genome", help="path to a genome JSON file")
    bench.add_argument("--split", choices=["train", "holdout", "all"], default="train")
    bench.add_argument("--backend", choices=["sim"], default="sim")
    bench.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"run": _cmd_run, "show": _cmd_show, "bench": _cmd_bench}
    return handlers[args.command](args)


def _progress(record: GenerationRecord) -> None:
    best_id, best_fitness = record.members[0]
    print(
        f"generation {record.index}: best fitness {best_fitness:.4f} ({best_id}), "
        f"{len(record.members)} members",
        file=sys.stderr,
        flush=True,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    config = EvolutionConfig(
        population=args.population,
        generations=args.generations,
        seed=args.seed,
        cost_weight=args.cost_weight,
        eval_seeds=args.eval_seeds,
    )
    out = Path(args.out) if args.out else Path("runs") / f"{args.backend}-seed{args.seed}"

    if args.backend == "anthropic" and args.eval_seeds > 1:
        print(
            "error: --eval-seeds above 1 needs the simulated backend; repeating paid "
            "Anthropic calls would multiply the cost",
            file=sys.stderr,
        )
        return EXIT_USAGE

    try:
        if args.backend == "anthropic":
            print("warning: --backend anthropic calls the paid Anthropic API.", file=sys.stderr)
            if not args.yes:
                print("error: pass --yes to run against the paid API", file=sys.stderr)
                return EXIT_USAGE
            backend = make_backend("anthropic")
            architect = Architect(backend, random.Random(config.seed + 1))
            run = evolve(
                config,
                backend,
                load_benchmark("train"),
                load_benchmark("holdout"),
                architect,
                on_generation=_progress,
            )
        else:
            run = run_simulated(config, on_generation=_progress)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    out.mkdir(parents=True, exist_ok=True)
    (out / RUN_NAME).write_text(json.dumps(run.to_dict(), indent=2, sort_keys=True))
    report = render_markdown(run)
    (out / REPORT_NAME).write_text(report)
    sys.stdout.write(report)
    print(f"wrote {out / RUN_NAME} and {out / REPORT_NAME}", file=sys.stderr)
    return EXIT_OK


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
