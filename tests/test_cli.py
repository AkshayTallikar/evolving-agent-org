import json

import pytest

from evorg import cli
from evorg.cli import main
from evorg.genome import seed_genome
from evorg.report import render_markdown


def _run_tiny(tmp_path, name="r"):
    out = tmp_path / name
    code = main(["run", "--generations", "1", "--population", "3", "--out", str(out)])
    assert code == 0
    return out


def test_run_writes_run_json_and_report(tmp_path, capsys):
    out = _run_tiny(tmp_path)

    data = json.loads((out / "run.json").read_text())
    assert data["champion_id"] in data["genomes"]
    assert data["backend"] == "sim"

    report = (out / "report.md").read_text()
    assert report.startswith("# evorg run")
    assert "## Champion" in report
    assert data["champion_id"] in report

    assert capsys.readouterr().out == report


def test_run_prints_progress_to_stderr(tmp_path, capsys):
    _run_tiny(tmp_path)
    err = capsys.readouterr().err
    assert "generation 0:" in err
    assert "generation 1:" in err


def test_show_prints_saved_report(tmp_path, capsys):
    out = _run_tiny(tmp_path)
    capsys.readouterr()

    assert main(["show", str(out)]) == 0
    assert capsys.readouterr().out == (out / "report.md").read_text()


def test_show_missing_dir_returns_1(tmp_path, capsys):
    assert main(["show", str(tmp_path / "missing")]) == 1
    assert "no report" in capsys.readouterr().err


def test_bench_genome_file_prints_totals(tmp_path, capsys):
    path = tmp_path / "genome.json"
    path.write_text(json.dumps(seed_genome().to_dict()))

    assert main(["bench", str(path), "--split", "train"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("task")
    assert "Totals:" in out


def test_bench_invalid_genome_returns_1(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"roles": [], "max_retries": 1}))

    assert main(["bench", str(path)]) == 1
    assert "cannot load genome" in capsys.readouterr().err


def test_bench_missing_file_returns_1(tmp_path, capsys):
    assert main(["bench", str(tmp_path / "nope.json")]) == 1


def test_anthropic_without_yes_returns_2_and_builds_no_backend(tmp_path, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise AssertionError("make_backend must not be called without --yes")

    monkeypatch.setattr(cli, "make_backend", fail)
    code = main(["run", "--backend", "anthropic", "--out", str(tmp_path / "a")])

    assert code == 2
    assert "--yes" in capsys.readouterr().err
    assert not (tmp_path / "a").exists()


def test_report_module_renders_same_text_as_cli(tmp_path):
    from evorg.evolution import EvolutionConfig, run_simulated

    out = _run_tiny(tmp_path)
    run = run_simulated(EvolutionConfig(population=3, generations=1, seed=0))
    assert (out / "report.md").read_text() == render_markdown(run)


@pytest.mark.parametrize("argv", [[], ["unknown"]])
def test_bad_command_exits_with_usage_error(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_eval_seeds_flag_is_recorded_in_run_json(tmp_path):
    out = tmp_path / "multi"
    code = main(
        [
            "run",
            "--generations",
            "1",
            "--population",
            "3",
            "--eval-seeds",
            "2",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    data = json.loads((out / "run.json").read_text())
    assert data["config"]["eval_seeds"] == 2
    assert "scored over 2 simulator seeds" in (out / "report.md").read_text()


def test_eval_seeds_defaults_to_one(tmp_path):
    out = _run_tiny(tmp_path)
    data = json.loads((out / "run.json").read_text())
    assert data["config"]["eval_seeds"] == 1
    assert "simulator seeds" not in (out / "report.md").read_text().splitlines()[2]


def test_anthropic_with_several_eval_seeds_returns_2(tmp_path, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise AssertionError("make_backend must not be called for eval_seeds > 1")

    monkeypatch.setattr(cli, "make_backend", fail)
    code = main(
        [
            "run",
            "--backend",
            "anthropic",
            "--yes",
            "--eval-seeds",
            "2",
            "--out",
            str(tmp_path / "a"),
        ]
    )

    assert code == 2
    assert "--eval-seeds" in capsys.readouterr().err
    assert not (tmp_path / "a").exists()


def test_crossover_rate_flag_is_recorded_in_run_json(tmp_path):
    out = tmp_path / "xo"
    argv = ["run", "--generations", "1", "--population", "3", "--out", str(out)]
    code = main([*argv, "--crossover-rate", "0.3"])
    assert code == 0
    data = json.loads((out / "run.json").read_text())
    assert data["config"]["crossover_rate"] == 0.3
    assert "crossover rate 0.3" in (out / "report.md").read_text()


def test_crossover_rate_defaults_to_zero_and_is_not_in_the_report(tmp_path):
    out = _run_tiny(tmp_path)
    data = json.loads((out / "run.json").read_text())
    assert data["config"]["crossover_rate"] == 0.0
    assert "crossover rate" not in (out / "report.md").read_text()


def test_crossover_rate_out_of_range_returns_1(tmp_path, capsys):
    argv = ["run", "--generations", "1", "--population", "3", "--out", str(tmp_path / "bad")]
    code = main([*argv, "--crossover-rate", "2"])
    assert code == 1
    assert "crossover_rate" in capsys.readouterr().err


def test_selection_flag_is_recorded_in_run_json_and_report(tmp_path):
    out = tmp_path / "pareto"
    argv = ["run", "--generations", "1", "--population", "3", "--out", str(out)]
    assert main([*argv, "--selection", "pareto"]) == 0
    data = json.loads((out / "run.json").read_text())
    assert data["config"]["selection"] == "pareto"
    assert "pareto selection" in (out / "report.md").read_text()


def test_selection_defaults_to_scalar(tmp_path):
    out = _run_tiny(tmp_path)
    assert json.loads((out / "run.json").read_text())["config"]["selection"] == "scalar"


def test_selection_flag_rejects_unknown_mode(tmp_path, capsys):
    argv = ["run", "--generations", "1", "--population", "3", "--out", str(tmp_path / "bad")]
    with pytest.raises(SystemExit) as excinfo:
        main([*argv, "--selection", "nsga"])
    assert excinfo.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
