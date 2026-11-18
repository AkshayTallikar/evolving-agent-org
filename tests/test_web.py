"""Tests for the lineage explorer API, the static page and the ``evorg serve`` command."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from evorg.cli import EXIT_USAGE, build_parser, main  # noqa: E402
from evorg.web import create_app  # noqa: E402
from evorg.web.app import INDEX_FILE, champion_path  # noqa: E402

RUN_ARGS = ["run", "--generations", "3", "--population", "4", "--seed", "3"]


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    """A runs directory holding one real run written by the ``evorg run`` command."""
    root = tmp_path / "runs"
    assert main([*RUN_ARGS, "--out", str(root / "tiny")]) == 0
    return root


@pytest.fixture
def client(runs_dir: Path) -> TestClient:
    return TestClient(create_app(runs_dir))


@pytest.fixture
def run_json(runs_dir: Path) -> dict[str, Any]:
    return json.loads((runs_dir / "tiny" / "run.json").read_text())


def _ancestors(genomes: dict[str, Any], gid: str) -> set[str]:
    """Independent check: walk parent links from ``gid`` and return it plus its ancestors."""
    seen: set[str] = set()
    stack = [gid]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(genomes[cur]["parent_ids"])
    return seen


def test_list_runs_summarises_each_run(client: TestClient, run_json: dict[str, Any]) -> None:
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    rows = resp.json()
    assert [row["name"] for row in rows] == ["tiny"]
    row = rows[0]
    assert row["backend"] == "sim"
    assert row["seed"] == 3
    assert row["population"] == 4
    assert row["generations"] == 3
    assert row["champion_id"] == run_json["champion_id"]
    champion_fitness = run_json["evaluations"][run_json["champion_id"]]["fitness"]
    assert row["best_fitness"] == pytest.approx(champion_fitness)


def test_list_runs_skips_folders_without_run_json(runs_dir: Path, client: TestClient) -> None:
    (runs_dir / "empty").mkdir()
    (runs_dir / "broken").mkdir()
    (runs_dir / "broken" / "run.json").write_text("{not json")
    (runs_dir / "loose.txt").write_text("not a run")
    assert [row["name"] for row in client.get("/api/runs").json()] == ["tiny"]


def test_list_runs_with_missing_root_is_empty(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "absent"))
    assert client.get("/api/runs").json() == []


def test_get_run_returns_run_json_and_tree(
    client: TestClient, run_json: dict[str, Any]
) -> None:
    resp = client.get("/api/runs/tiny")
    assert resp.status_code == 200
    body = resp.json()
    for key, value in run_json.items():
        assert body[key] == value
    tree = {node["id"]: node for node in body["tree"]}
    assert set(tree) == set(run_json["genomes"])
    for gid, node in tree.items():
        genome = run_json["genomes"][gid]
        assert node["generation"] == genome["generation"]
        assert node["parents"] == genome["parent_ids"]
        assert node["note"] == genome["mutation_note"]
        assert node["fitness"] == run_json["evaluations"][gid]["fitness"]
        assert node["pass_rate"] == run_json["evaluations"][gid]["pass_rate"]
        assert node["mean_cost_usd"] == run_json["evaluations"][gid]["mean_cost_usd"]
        assert node["is_champion"] == (gid == run_json["champion_id"])


def test_champion_path_flags_champion_and_all_ancestors(
    client: TestClient, run_json: dict[str, Any]
) -> None:
    body = client.get("/api/runs/tiny").json()
    flagged = {node["id"] for node in body["tree"] if node["on_champion_path"]}
    expected = _ancestors(run_json["genomes"], run_json["champion_id"])
    assert flagged == expected
    assert run_json["champion_id"] in flagged
    outsiders = set(run_json["genomes"]) - expected
    assert outsiders, "the tiny run needs at least one genome off the champion's path"
    assert not flagged & outsiders


def test_champion_path_helper_handles_missing_champion() -> None:
    assert champion_path({}, "nope") == set()
    assert champion_path({"a": {"parent_ids": ["gone"]}}, "a") == {"a"}


def test_tree_drops_parents_missing_from_run_json() -> None:
    from evorg.web.app import build_tree

    data = {
        "champion_id": "b",
        "genomes": {
            "b": {"parent_ids": ["a", "ghost"], "generation": 1, "mutation_note": ""},
            "a": {"parent_ids": [], "generation": 0, "mutation_note": ""},
        },
        "evaluations": {},
    }
    nodes = {node["id"]: node for node in build_tree(data)}
    assert nodes["b"]["parents"] == ["a"]
    assert nodes["b"]["fitness"] is None
    assert nodes["a"]["on_champion_path"] is True


def test_get_genome_returns_genome_evaluation_and_diff(
    client: TestClient, run_json: dict[str, Any]
) -> None:
    from evorg.genome import OrgGenome, diff_genomes

    child_id = next(gid for gid, g in run_json["genomes"].items() if g["parent_ids"])
    parent_id = run_json["genomes"][child_id]["parent_ids"][0]
    resp = client.get(f"/api/runs/tiny/genomes/{child_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["genome_id"] == child_id
    assert body["genome"] == run_json["genomes"][child_id]
    assert body["evaluation"] == run_json["evaluations"][child_id]
    expected = diff_genomes(
        OrgGenome.from_dict(run_json["genomes"][parent_id]),
        OrgGenome.from_dict(run_json["genomes"][child_id]),
    )
    assert body["diff_from_parent"] == expected


def test_get_genome_without_parent_has_empty_diff(
    client: TestClient, run_json: dict[str, Any]
) -> None:
    root_id = next(gid for gid, g in run_json["genomes"].items() if not g["parent_ids"])
    body = client.get(f"/api/runs/tiny/genomes/{root_id}").json()
    assert body["diff_from_parent"] == []


def test_get_genome_with_unrecorded_parent_has_empty_diff(runs_dir: Path) -> None:
    path = runs_dir / "tiny" / "run.json"
    data = json.loads(path.read_text())
    gid = next(g for g, v in data["genomes"].items() if v["parent_ids"])
    del data["genomes"][data["genomes"][gid]["parent_ids"][0]]
    path.write_text(json.dumps(data))
    client = TestClient(create_app(runs_dir))
    assert client.get(f"/api/runs/tiny/genomes/{gid}").json()["diff_from_parent"] == []


def test_unknown_run_is_404_with_json_detail(client: TestClient) -> None:
    resp = client.get("/api/runs/missing")
    assert resp.status_code == 404
    assert "detail" in resp.json()


def test_unknown_genome_is_404_with_json_detail(client: TestClient) -> None:
    resp = client.get("/api/runs/tiny/genomes/0000000000")
    assert resp.status_code == 404
    assert "detail" in resp.json()


def test_run_folder_without_run_json_is_404(runs_dir: Path, client: TestClient) -> None:
    (runs_dir / "bare").mkdir()
    assert client.get("/api/runs/bare").status_code == 404


def test_unreadable_run_json_is_404(runs_dir: Path, client: TestClient) -> None:
    (runs_dir / "bad").mkdir()
    (runs_dir / "bad" / "run.json").write_text("[]")
    resp = client.get("/api/runs/bad")
    assert resp.status_code == 404
    assert "readable" in resp.json()["detail"]


@pytest.mark.parametrize(
    "name",
    ["%2E%2E", "%2E", ".hidden", "a..b", "..%2F..%2Fetc", "%2e%2e", "x%5Cy"],
)
def test_traversal_names_are_rejected(client: TestClient, name: str) -> None:
    resp = client.get(f"/api/runs/{name}")
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/json")


def test_symlink_out_of_runs_dir_is_rejected(tmp_path: Path, runs_dir: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "run.json").write_text(json.dumps({"genomes": {}}))
    (runs_dir / "escape").symlink_to(outside, target_is_directory=True)
    client = TestClient(create_app(runs_dir))
    assert client.get("/api/runs/escape").status_code == 404


def test_traversal_guard_blocks_file_outside_runs_dir(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "run.json").write_text("{}")
    client = TestClient(create_app(root))
    assert client.get("/api/runs/..%2Fsecret").status_code == 404


def test_index_is_served_with_pinned_react(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "react@18.3.1" in resp.text
    assert resp.text == INDEX_FILE.read_text()


def test_serve_parses_defaults_to_localhost() -> None:
    args = build_parser().parse_args(["serve"])
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.runs_dir == "runs"


def test_serve_calls_uvicorn_on_localhost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []
    fake = types.ModuleType("uvicorn")
    fake.run = lambda app, **kwargs: calls.append({"app": app, **kwargs})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", fake)

    assert main(["serve", "--runs-dir", str(tmp_path), "--port", "9001"]) == 0
    assert len(calls) == 1
    assert calls[0]["host"] == "127.0.0.1"
    assert calls[0]["port"] == 9001
    assert calls[0]["app"].title == "evorg lineage explorer"


def test_serve_without_uvicorn_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    assert main(["serve"]) == EXIT_USAGE
    assert 'install the web extra: pip install -e ".[web]"' in capsys.readouterr().err
