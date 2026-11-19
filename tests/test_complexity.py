"""The optional parsimony penalty: fitness, validation, the CLI flag, saved config and ablation."""

import json
import math

import pytest

from evorg.ablation import ablate, complexity_weight_of
from evorg.bench import load_benchmark, task_index
from evorg.cli import main
from evorg.evolution import EvolutionConfig, _config_dict, evaluate, run_simulated
from evorg.genome import OrgGenome, RoleSpec, seed_genome
from evorg.llm.simulated import SimulatedBackend


def _bloated() -> OrgGenome:
    """Planner (2 instructions), coder (3), reviewer (1): 6 instructions over 3 roles."""
    return OrgGenome(
        roles=(
            RoleSpec("planner", "planner", ("Make a short plan.", "Number the steps.")),
            RoleSpec(
                "coder",
                "coder",
                ("Write a correct Python function.", "Use type hints.", "Handle empty input."),
            ),
            RoleSpec("reviewer", "reviewer", ("Look for off-by-one errors.",)),
        ),
        max_retries=1,
    )


def test_size_counts_instructions_over_all_roles_plus_extra_roles():
    assert seed_genome().size() == 1  # one coder, one instruction, no extra role
    assert _bloated().size() == 6 + 2  # six instructions, two roles beyond the first


def test_fitness_subtracts_weight_times_size_and_leaves_pass_rate_and_cost_alone():
    tasks = load_benchmark("train")[:4]
    genome = _bloated()
    backend = SimulatedBackend(task_index(), seed=0)

    plain = evaluate(genome, tasks, backend, 10.0)
    penalised = evaluate(genome, tasks, backend, 10.0, complexity_weight=0.05)

    assert penalised.pass_rate == plain.pass_rate
    assert penalised.mean_cost_usd == plain.mean_cost_usd
    expected = plain.pass_rate - 10.0 * plain.mean_cost_usd - 0.05 * genome.size()
    assert penalised.fitness == pytest.approx(expected)
    assert penalised.fitness < plain.fitness


def test_default_weight_is_zero_and_fitness_is_exactly_the_old_formula():
    assert EvolutionConfig().complexity_weight == 0.0
    tasks = load_benchmark("train")[:4]
    backend = SimulatedBackend(task_index(), seed=0)
    ev = evaluate(seed_genome(), tasks, backend, 10.0)
    assert ev.fitness == ev.pass_rate - 10.0 * ev.mean_cost_usd  # bit-for-bit, not approx


def test_config_dict_leaves_the_weight_out_at_zero_and_writes_it_when_set():
    assert "complexity_weight" not in _config_dict(EvolutionConfig())
    assert _config_dict(EvolutionConfig(complexity_weight=0.01))["complexity_weight"] == 0.01


@pytest.mark.parametrize("weight", [-0.01, float("nan"), float("inf"), -float("inf")])
def test_complexity_weight_must_be_finite_and_non_negative(weight):
    with pytest.raises(ValueError, match="complexity_weight"):
        run_simulated(EvolutionConfig(population=4, generations=1, complexity_weight=weight))


def test_cli_flag_is_recorded_in_run_json_and_report(tmp_path):
    out = tmp_path / "parsimony"
    argv = ["run", "--generations", "1", "--population", "3", "--out", str(out)]
    assert main([*argv, "--complexity-weight", "0.02"]) == 0
    data = json.loads((out / "run.json").read_text())
    assert data["config"]["complexity_weight"] == 0.02
    assert "complexity weight 0.02" in (out / "report.md").read_text()


def test_cli_default_leaves_no_trace_in_run_json_or_report(tmp_path):
    out = tmp_path / "default"
    main(["run", "--generations", "1", "--population", "3", "--out", str(out)])
    data = json.loads((out / "run.json").read_text())
    assert "complexity_weight" not in data["config"]
    assert "complexity weight" not in (out / "report.md").read_text()


def test_cli_rejects_a_negative_weight(tmp_path, capsys):
    argv = ["run", "--generations", "1", "--population", "3", "--out", str(tmp_path / "bad")]
    assert main([*argv, "--complexity-weight", "-1"]) == 1
    assert "complexity_weight" in capsys.readouterr().err


def test_complexity_weight_of_reads_run_json_and_defaults_to_zero():
    assert complexity_weight_of({"config": {"cost_weight": 10.0}}) == 0.0
    assert complexity_weight_of({"config": {"complexity_weight": 0.03}}) == 0.03


def test_ablation_fitness_honours_the_run_weight():
    tasks = load_benchmark("train")[:4]
    champion = _bloated()
    seed = seed_genome()
    plain = ablate(champion, seed, tasks, [0], 10.0)
    weighted = ablate(champion, seed, tasks, [0], 10.0, complexity_weight=0.5)

    assert weighted.champion_pass_rate == plain.champion_pass_rate
    assert math.isclose(
        weighted.champion_fitness, plain.champion_fitness - 0.5 * champion.size(), abs_tol=1e-9
    )
