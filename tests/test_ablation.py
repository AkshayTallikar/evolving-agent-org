import json

import pytest

from evorg import ablation
from evorg.ablation import (
    AblationResult,
    AblationRow,
    Delta,
    Variant,
    ablate,
    build_variants,
    champion_and_seed,
    render_ablation,
    seed_offsets,
)
from evorg.bench import load_benchmark, task_index
from evorg.cli import main
from evorg.evolution import HOLDOUT_SEED_OFFSET, TRAIN_SEED_STEP, evaluate
from evorg.genome import OrgGenome, RoleSpec, diff_genomes, seed_genome
from evorg.llm.simulated import SimulatedBackend


def _rich_champion() -> OrgGenome:
    """Planner (small), coder (large, two instructions), reviewer (small), retries 2."""
    return OrgGenome(
        roles=(
            RoleSpec("planner", "planner", ("Make a short plan.",), "small"),
            RoleSpec(
                "coder", "coder", ("Write a correct Python function.", "Use type hints."), "large"
            ),
            RoleSpec("reviewer", "reviewer", (), "small"),
        ),
        max_retries=2,
    )


def test_variants_cover_each_gene_type():
    champion = _rich_champion()
    variants, skipped = build_variants(champion, seed_genome())
    names = [v.name for v in variants]

    assert names == [
        "role:planner",
        "instruction:planner[0]",
        "instruction:coder[0]",
        "instruction:coder[1]",
        "model:coder",
        "role:reviewer",
        "retries",
    ]
    assert [s.name for s in skipped] == ["role:coder"]
    kinds = {v.name: v.kind for v in variants}
    assert kinds["role:planner"] == "role"
    assert kinds["instruction:coder[1]"] == "instruction"
    assert kinds["model:coder"] == "model"
    assert kinds["retries"] == "retries"


def test_removing_the_only_coder_is_skipped_with_reason():
    variants, skipped = build_variants(seed_genome(), seed_genome())
    assert all(v.name != "role:coder" for v in variants)
    assert len(skipped) == 1
    assert skipped[0].name == "role:coder"
    assert "exactly one" in skipped[0].reason


def test_each_variant_differs_from_champion_by_exactly_one_gene():
    champion = _rich_champion()
    variants, _ = build_variants(champion, seed_genome())
    assert variants
    for variant in variants:
        assert len(diff_genomes(champion, variant.genome)) == 1, variant.name
        variant.genome.validate()


def test_gene_absent_from_champion_is_not_produced():
    # The champion is the seed itself: no model above small, retries equal to the seed's,
    # no planner, tester or reviewer. Only the coder's one instruction can be ablated.
    champion = seed_genome()
    variants, skipped = build_variants(champion, seed_genome())
    assert [v.name for v in variants] == ["instruction:coder[0]"]
    assert [s.name for s in skipped] == ["role:coder"]


def test_retry_variant_uses_seed_value_and_is_absent_when_equal():
    champion = _rich_champion()
    seed = seed_genome()  # retries 1
    variants, _ = build_variants(champion, seed)
    retry = next(v for v in variants if v.name == "retries")
    assert retry.genome.max_retries == seed.max_retries == 1

    same, _ = build_variants(seed_genome(), seed_genome())
    assert all(v.name != "retries" for v in same)


def test_model_variant_reverts_to_small():
    variants, _ = build_variants(_rich_champion(), seed_genome())
    model = next(v for v in variants if v.name == "model:coder")
    assert model.genome.role("coder").model == "small"
    assert model.genome.role("coder").instructions == _rich_champion().role("coder").instructions


def test_seed_offsets_keep_holdout_apart_from_training():
    train = seed_offsets(0, "train", 3)
    holdout = seed_offsets(0, "holdout", 3)
    assert train == [0, TRAIN_SEED_STEP, 2 * TRAIN_SEED_STEP]
    assert holdout == [HOLDOUT_SEED_OFFSET + TRAIN_SEED_STEP * k for k in range(3)]
    assert not set(train) & set(holdout)


def test_ablate_rejects_empty_inputs():
    tasks = load_benchmark("holdout")[:1]
    with pytest.raises(ValueError, match="seed"):
        ablate(seed_genome(), seed_genome(), tasks, [], 10.0)
    with pytest.raises(ValueError, match="task"):
        ablate(seed_genome(), seed_genome(), [], [0], 10.0)


def test_ablate_is_deterministic_and_pairs_deltas():
    tasks = load_benchmark("holdout")[:3]
    champion = _rich_champion()
    seed = seed_genome()
    seeds = [0, TRAIN_SEED_STEP]

    first = ablate(champion, seed, tasks, seeds, 10.0, split="holdout")
    second = ablate(champion, seed, tasks, seeds, 10.0, split="holdout")
    assert first == second

    # A row's pass-rate delta is the variant's mean pass rate minus the champion's, computed
    # directly from the same (seed, task) backends.
    index = task_index()
    champ_pass = [
        evaluate(champion, tasks, SimulatedBackend(index, seed=s), 10.0).pass_rate for s in seeds
    ]
    row = next(r for r in first.rows if r.variant.name == "role:reviewer")
    var_pass = [
        evaluate(row.variant.genome, tasks, SimulatedBackend(index, seed=s), 10.0).pass_rate
        for s in seeds
    ]
    expected = sum(v - c for v, c in zip(var_pass, champ_pass, strict=True)) / len(seeds)
    assert row.pass_rate.mean == pytest.approx(expected)
    assert row.pass_rate.low <= row.pass_rate.mean <= row.pass_rate.high


def test_rows_sorted_most_harmful_first():
    tasks = load_benchmark("holdout")[:3]
    result = ablate(_rich_champion(), seed_genome(), tasks, [0, TRAIN_SEED_STEP], 10.0)
    fitness = [r.fitness.mean for r in result.rows]
    assert fitness == sorted(fitness)


def test_champion_and_seed_reads_run_dict_and_refuses_paid_backend():
    champion = _rich_champion()
    seed = seed_genome()
    data = {
        "backend": "sim",
        "champion_id": champion.genome_id,
        "seed_id": seed.genome_id,
        "genomes": {champion.genome_id: champion.to_dict(), seed.genome_id: seed.to_dict()},
        "config": {"cost_weight": 7.5},
    }
    got_champion, got_seed, weight = champion_and_seed(data)
    assert got_champion.genome_id == champion.genome_id
    assert got_seed.genome_id == seed.genome_id
    assert weight == 7.5

    data["backend"] = "anthropic"
    with pytest.raises(ValueError, match="simulated backend"):
        champion_and_seed(data)


def _row(name: str, pass_mean: float, fitness_mean: float) -> AblationRow:
    genome = seed_genome()
    return AblationRow(
        variant=Variant(
            name=f"role:{name}", kind="role", label=f"removing the {name}", genome=genome
        ),
        pass_rate=Delta(pass_mean, pass_mean - 0.04, pass_mean + 0.04),
        cost=Delta(-0.0002, -0.0003, -0.0001),
        fitness=Delta(fitness_mean, fitness_mean - 0.01, fitness_mean + 0.01),
        verdict=ablation._verdict(
            f"removing the {name}",
            Delta(pass_mean, pass_mean - 0.04, pass_mean + 0.04),
            Delta(-0.0002, -0.0003, -0.0001),
            Delta(fitness_mean, fitness_mean - 0.01, fitness_mean + 0.01),
            0.001,
        ),
    )


def test_render_markdown_has_table_verdicts_and_skipped_section():
    result = AblationResult(
        champion_id="abc1234567",
        split="holdout",
        seeds=(500000, 501000),
        n_tasks=12,
        champion_pass_rate=0.75,
        champion_cost=0.001,
        champion_fitness=0.65,
        rows=(_row("reviewer", -0.125, -0.2), _row("tester", 0.0, 0.01)),
        skipped=(
            ablation.Skipped(
                name="role:coder",
                label="removing the coder",
                reason="the coder cannot be removed",
            ),
        ),
    )
    text = render_ablation(result)
    assert text.startswith("# evorg ablation")
    assert "| gene | pass rate change |" in text
    assert "| role:reviewer |" in text
    assert "Removing the reviewer costs 12 points of pass rate (95% CI" in text
    assert "saves 20% cost" in text
    assert "Keeping it is worth its cost" in text
    assert "leaves pass rate about unchanged" in text
    assert text.index("role:reviewer") < text.index("role:tester")
    assert "## Skipped variants" in text
    assert "role:coder: removing the coder, skipped because" in text
    assert "simulated backend" in text


def test_render_without_variants_says_so():
    result = AblationResult(
        champion_id="abc",
        split="train",
        seeds=(0,),
        n_tasks=1,
        champion_pass_rate=1.0,
        champion_cost=0.0,
        champion_fitness=1.0,
        rows=(),
        skipped=(),
    )
    text = render_ablation(result)
    assert "| (none) |" in text
    assert "None. Every gene's removal gives a valid genome." in text


def test_cli_ablate_writes_markdown_for_a_saved_run(tmp_path, capsys):
    run_dir = tmp_path / "run"
    assert main(["run", "--generations", "1", "--population", "3", "--out", str(run_dir)]) == 0
    capsys.readouterr()

    out_file = tmp_path / "ablation.md"
    code = main(["ablate", str(run_dir), "--seeds", "1", "--out", str(out_file)])
    assert code == 0
    text = out_file.read_text()
    assert text.startswith("# evorg ablation")
    assert "on the holdout split" in text
    assert json.loads((run_dir / "run.json").read_text())["champion_id"] in text


def test_cli_ablate_prints_to_stdout_by_default(tmp_path, capsys):
    run_dir = tmp_path / "run"
    assert main(["run", "--generations", "0", "--population", "1", "--out", str(run_dir)]) == 0
    capsys.readouterr()

    assert main(["ablate", str(run_dir), "--seeds", "1", "--split", "train"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# evorg ablation")
    assert "on the train split" in out


def test_cli_ablate_missing_run_returns_1(tmp_path, capsys):
    assert main(["ablate", str(tmp_path / "missing")]) == 1
    assert "cannot load run" in capsys.readouterr().err


def test_cli_ablate_rejects_zero_seeds(tmp_path, capsys):
    assert main(["ablate", str(tmp_path), "--seeds", "0"]) == 2
    assert "--seeds" in capsys.readouterr().err
