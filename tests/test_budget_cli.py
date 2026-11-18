"""CLI behaviour of the spending cap and the cost estimate, with a fake Anthropic client."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from evorg import cli
from evorg.cli import main
from evorg.llm import AnthropicBackend, count_tokens_approx
from evorg.llm.anthropic_backend import MAX_TOKENS


class FakeMessages:
    """Returns a short text with usage that matches the request, scaled by ``input_scale``."""

    def __init__(self, input_scale: float = 1.0) -> None:
        self.input_scale = input_scale
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        approx = count_tokens_approx(kwargs["system"] + kwargs["messages"][0]["content"])
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(
                input_tokens=int(approx * self.input_scale), output_tokens=MAX_TOKENS
            ),
            stop_reason="end_turn",
        )


class FakeClient:
    def __init__(self, input_scale: float = 1.0) -> None:
        self.messages = FakeMessages(input_scale)


def _patch_fake_client(monkeypatch, input_scale: float = 1.0) -> FakeClient:
    client = FakeClient(input_scale)

    def fake_make_backend(name, **kwargs):
        assert name == "anthropic"
        return AnthropicBackend(client=client)

    monkeypatch.setattr(cli, "make_backend", fake_make_backend)
    return client


def _no_client(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("no client may be built in this case")

    monkeypatch.setattr(cli, "make_backend", fail)


def test_anthropic_without_max_usd_exits_2_before_building_a_client(tmp_path, monkeypatch, capsys):
    _no_client(monkeypatch)
    code = main(["run", "--backend", "anthropic", "--yes", "--out", str(tmp_path / "a")])
    assert code == 2
    assert "--max-usd" in capsys.readouterr().err
    assert not (tmp_path / "a").exists()


@pytest.mark.parametrize("value", ["0", "-1", "nan"])
def test_non_positive_max_usd_exits_2(tmp_path, monkeypatch, capsys, value):
    _no_client(monkeypatch)
    argv = ["run", "--backend", "anthropic", "--yes", "--max-usd", value, "--out", str(tmp_path)]
    assert main(argv) == 2
    assert "--max-usd" in capsys.readouterr().err


def test_max_usd_with_sim_backend_exits_2(tmp_path, capsys):
    code = main(["run", "--max-usd", "5", "--out", str(tmp_path / "s")])
    assert code == 2
    assert "--max-usd" in capsys.readouterr().err


def test_small_anthropic_run_with_cap_succeeds_and_reports_spend(tmp_path, monkeypatch, capsys):
    _patch_fake_client(monkeypatch)
    out = tmp_path / "real"
    code = main(
        [
            "run",
            "--backend",
            "anthropic",
            "--yes",
            "--max-usd",
            "5",
            "--generations",
            "1",
            "--population",
            "2",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert (out / "run.json").is_file()
    err = capsys.readouterr().err
    match = re.search(r"spent \$([0-9.]+) of \$5.00 cap across (\d+) calls", err)
    assert match is not None
    assert 0 < float(match.group(1)) < 5
    assert int(match.group(2)) > 0


def test_architect_spend_is_part_of_the_total(tmp_path, monkeypatch, capsys):
    _patch_fake_client(monkeypatch)
    out = tmp_path / "arch"
    code = main(
        [
            "run",
            "--backend",
            "anthropic",
            "--yes",
            "--max-usd",
            "5",
            "--generations",
            "1",
            "--population",
            "2",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    spent = float(re.search(r"spent \$([0-9.]+)", capsys.readouterr().err).group(1))
    data = json.loads((out / "run.json").read_text())
    executor_cost = sum(
        t["cost_usd"]
        for ev in [*data["evaluations"].values(), *data["holdout"].values()]
        for t in ev["per_task"]
    )
    architect_cost = data["architect_stats"]["cost_usd"]
    assert architect_cost > 0
    assert spent > executor_cost
    assert spent == pytest.approx(executor_cost + architect_cost, abs=2e-4)


def test_expensive_calls_trigger_exit_3(tmp_path, monkeypatch, capsys):
    _patch_fake_client(monkeypatch, input_scale=1000.0)  # actual cost far above the worst case
    out = tmp_path / "costly"
    code = main(
        [
            "run",
            "--backend",
            "anthropic",
            "--yes",
            "--max-usd",
            "1",
            "--generations",
            "1",
            "--population",
            "2",
            "--out",
            str(out),
        ]
    )
    assert code == 3
    err = capsys.readouterr().err
    assert re.search(r"stopped: spending cap reached \(\$[0-9.]+ of \$1.00\)", err)
    assert not (out / "run.json").exists()


def test_estimate_prints_calls_and_dollar_bound(capsys):
    code = main(["estimate", "--generations", "1", "--population", "3"])
    assert code == 0
    out = capsys.readouterr().out
    assert "LLM calls" in out
    assert re.search(r"^small\s+\d+\s+\d+\.\d{6}$", out, re.MULTILINE)
    assert re.search(r"^total\s+\d+\s+\d+\.\d{6}$", out, re.MULTILINE)
    assert "Upper bound for a real run: $" in out
    assert "stand in" in out


def test_estimate_with_large_share_reprices_the_mix(capsys):
    assert main(["estimate", "--generations", "1", "--population", "3", "--large-share", "1"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"^small\s+0\s+", out, re.MULTILINE)


def test_estimate_refuses_several_eval_seeds(capsys):
    assert main(["estimate", "--eval-seeds", "2"]) == 2
    assert "--eval-seeds" in capsys.readouterr().err


def test_estimate_writes_nothing_and_makes_no_client(tmp_path, monkeypatch, capsys):
    _no_client(monkeypatch)
    before = set(tmp_path.iterdir())
    monkeypatch.chdir(tmp_path)
    assert main(["estimate", "--generations", "1", "--population", "2"]) == 0
    assert set(tmp_path.iterdir()) == before
