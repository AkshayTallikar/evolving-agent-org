"""Parallel evaluation: same results for any worker count, validation, and the CLI flag."""

import json
import time

import pytest

from evorg.bench import load_benchmark, task_index
from evorg.cli import build_parser, main
from evorg.evolution import EvolutionConfig, evaluate, evaluate_many, run_simulated
from evorg.genome import OrgGenome, RoleSpec, seed_genome
from evorg.llm.base import LLMRequest, LLMResponse
from evorg.llm.simulated import SimulatedBackend

SLOW_CALL_SECONDS = 0.05


class SlowBackend:
    """Sleeps per call to stand in for API latency. It has no state, so it is thread-safe."""

    name = "slow"

    def complete(self, request: LLMRequest) -> LLMResponse:
        time.sleep(SLOW_CALL_SECONDS)
        return LLMResponse(
            text="```python\ndef solve():\n    return None\n```",
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
        )


def _rich_genome() -> OrgGenome:
    """A genome with every role kind, so feedback loops and retries are exercised."""
    roles = (
        RoleSpec(name="planner", kind="planner", instructions=("Think about edge cases.",)),
        RoleSpec(
            name="coder",
            kind="coder",
            instructions=(
                "Write a correct Python function.",
                "Check the empty input case step by step.",
            ),
        ),
        RoleSpec(name="tester", kind="tester"),
        RoleSpec(name="reviewer", kind="reviewer", model="large", instructions=("Be careful.",)),
    )
    genome = OrgGenome(roles=roles, max_retries=2, mutation_note="rich test genome")
    genome.validate()
    return genome


def test_default_workers_is_one_and_not_saved_in_run_json():
    assert EvolutionConfig().workers == 1
    data = run_simulated(EvolutionConfig(population=2, generations=0)).to_dict()
    assert "workers" not in data["config"]


def test_evaluate_with_four_workers_equals_serial_for_seed_and_rich_genome():
    backend = SimulatedBackend(task_index(), seed=0)
    tasks = load_benchmark("train")
    for genome in (seed_genome(), _rich_genome()):
        serial = evaluate(genome, tasks, backend, 10.0, workers=1)
        parallel = evaluate(genome, tasks, backend, 10.0, workers=4)
        assert parallel == serial
        assert [r.task_id for r in parallel.results] == [t.id for t in tasks]


def test_evaluate_many_with_four_workers_equals_serial():
    index = task_index()
    backends = [SimulatedBackend(index, seed=0), SimulatedBackend(index, seed=1000)]
    tasks = load_benchmark("train")[:5]
    genome = _rich_genome()

    serial = evaluate_many(genome, tasks, backends, 10.0, workers=1)
    parallel = evaluate_many(genome, tasks, backends, 10.0, workers=4)
    assert parallel == serial
    assert parallel.results == tuple(
        r for b in backends for r in evaluate(genome, tasks, b, 10.0).results
    )


def test_run_simulated_with_three_workers_has_the_same_to_dict():
    small = {"population": 4, "generations": 2, "seed": 0}
    serial = run_simulated(EvolutionConfig(**small, workers=1)).to_dict()
    parallel = run_simulated(EvolutionConfig(**small, workers=3)).to_dict()
    assert json.dumps(parallel, sort_keys=True) == json.dumps(serial, sort_keys=True)


@pytest.mark.parametrize("workers", [0, -1])
def test_workers_below_one_is_rejected(workers):
    backend = SimulatedBackend(task_index(), seed=0)
    tasks = load_benchmark("train")[:1]
    with pytest.raises(ValueError, match="workers"):
        evaluate(seed_genome(), tasks, backend, 10.0, workers=workers)
    with pytest.raises(ValueError, match="workers"):
        evaluate_many(seed_genome(), tasks, [backend], 10.0, workers=workers)
    with pytest.raises(ValueError, match="workers"):
        run_simulated(EvolutionConfig(population=2, generations=0, workers=workers))


def test_four_workers_finish_clearly_faster_than_one():
    tasks = load_benchmark("train")
    assert len(tasks) == 8
    genome = seed_genome()
    backend = SlowBackend()

    start = time.perf_counter()
    serial = evaluate(genome, tasks, backend, 10.0, workers=1)
    serial_seconds = time.perf_counter() - start

    start = time.perf_counter()
    parallel = evaluate(genome, tasks, backend, 10.0, workers=4)
    parallel_seconds = time.perf_counter() - start

    assert parallel == serial
    assert parallel_seconds < 0.6 * serial_seconds


def test_cli_workers_flag_defaults_to_one_and_gives_same_run_json(tmp_path, capsys):
    assert build_parser().parse_args(["run"]).workers == 1
    assert build_parser().parse_args(["run", "--workers", "4"]).workers == 4

    base = ["run", "--generations", "1", "--population", "3"]
    assert main([*base, "--out", str(tmp_path / "one")]) == 0
    assert main([*base, "--workers", "3", "--out", str(tmp_path / "three")]) == 0
    serial = (tmp_path / "one" / "run.json").read_text()
    parallel = (tmp_path / "three" / "run.json").read_text()
    assert serial == parallel
    capsys.readouterr()


def test_cli_rejects_zero_workers(tmp_path, capsys):
    code = main(["run", "--workers", "0", "--out", str(tmp_path / "x")])
    assert code == 1
    assert "workers" in capsys.readouterr().err
