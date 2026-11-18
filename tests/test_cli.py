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
