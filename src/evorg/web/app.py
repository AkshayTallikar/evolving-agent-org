"""FastAPI app that serves saved evorg runs as JSON, plus one static lineage explorer page.

A run is any subdirectory of ``runs_dir`` that holds a ``run.json`` written by ``evorg run``.
Every response is computed from that file alone; nothing is written and no run is re-evolved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from evorg.genome import OrgGenome, diff_genomes

RUN_FILE = "run.json"
INDEX_FILE = Path(__file__).parent / "static" / "index.html"


def create_app(runs_dir: Path) -> FastAPI:
    """Build the app. Run directories are looked up under ``runs_dir`` at request time."""
    root = Path(runs_dir)
    app = FastAPI(title="evorg lineage explorer")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(INDEX_FILE, media_type="text/html")

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        """Summaries of every readable run, sorted by name."""
        summaries = []
        for path in sorted(root.iterdir()) if root.is_dir() else []:
            if not _is_plain_name(path.name) or not (path / RUN_FILE).is_file():
                continue
            try:
                data = _read_run(path / RUN_FILE)
            except (OSError, ValueError):
                continue
            summaries.append(run_summary(path.name, data))
        return summaries

    @app.get("/api/runs/{name}")
    def get_run(name: str) -> dict[str, Any]:
        """The run.json content plus a ``tree`` of nodes for the family view."""
        data = _load_run(root, name)
        return {**data, "tree": build_tree(data)}

    @app.get("/api/runs/{name}/genomes/{genome_id}")
    def get_genome(name: str, genome_id: str) -> dict[str, Any]:
        """One genome, its evaluation and the changes from its first parent."""
        data = _load_run(root, name)
        if genome_id not in data.get("genomes", {}):
            raise HTTPException(status_code=404, detail=f"genome {genome_id} not found in {name}")
        return genome_detail(data, genome_id)

    return app


def run_summary(name: str, data: dict[str, Any]) -> dict[str, Any]:
    """The fields the run picker needs, read from a run.json dict."""
    config = data.get("config", {})
    champion_id = data.get("champion_id")
    champion_ev = data.get("evaluations", {}).get(champion_id, {})
    return {
        "name": name,
        "backend": data.get("backend"),
        "seed": config.get("seed"),
        "population": config.get("population"),
        "generations": config.get("generations"),
        "champion_id": champion_id,
        "best_fitness": champion_ev.get("fitness"),
    }


def build_tree(data: dict[str, Any]) -> list[dict[str, Any]]:
    """One node per genome in run.json, ordered by generation then id.

    Parents that are not recorded in run.json are dropped, so every edge has both ends.
    ``on_champion_path`` is true for the champion and every genome it descends from.
    """
    genomes = data.get("genomes", {})
    evaluations = data.get("evaluations", {})
    champion_id = data.get("champion_id")
    on_path = champion_path(genomes, champion_id)
    nodes = []
    for gid, genome in sorted(genomes.items(), key=lambda kv: (kv[1].get("generation", 0), kv[0])):
        ev = evaluations.get(gid, {})
        nodes.append(
            {
                "id": gid,
                "parents": [p for p in genome.get("parent_ids", []) if p in genomes],
                "generation": genome.get("generation", 0),
                "fitness": ev.get("fitness"),
                "pass_rate": ev.get("pass_rate"),
                "mean_cost_usd": ev.get("mean_cost_usd"),
                "note": genome.get("mutation_note", ""),
                "is_champion": gid == champion_id,
                "on_champion_path": gid in on_path,
            }
        )
    return nodes


def champion_path(genomes: dict[str, Any], champion_id: str | None) -> set[str]:
    """The champion and all of its recorded ancestors, following every parent link."""
    if champion_id not in genomes:
        return set()
    path: set[str] = set()
    pending = [champion_id]
    while pending:
        gid = pending.pop()
        if gid in path or gid not in genomes:
            continue
        path.add(gid)
        pending.extend(genomes[gid].get("parent_ids", []))
    return path


def genome_detail(data: dict[str, Any], genome_id: str) -> dict[str, Any]:
    """The genome, its evaluation and ``diff_from_parent`` against its first recorded parent."""
    genomes = data["genomes"]
    raw = genomes[genome_id]
    diff: list[str] = []
    parent_ids = raw.get("parent_ids", [])
    if parent_ids and parent_ids[0] in genomes:
        parent = OrgGenome.from_dict(genomes[parent_ids[0]])
        diff = diff_genomes(parent, OrgGenome.from_dict(raw))
    return {
        "genome_id": genome_id,
        "genome": raw,
        "evaluation": data.get("evaluations", {}).get(genome_id),
        "diff_from_parent": diff,
    }


def _is_plain_name(name: str) -> bool:
    """A run name is one directory name: no separators, no ``..`` and no leading dot."""
    if not name or name.startswith(".") or ".." in name:
        return False
    return "/" not in name and "\\" not in name


def _load_run(root: Path, name: str) -> dict[str, Any]:
    if not _is_plain_name(name):
        raise HTTPException(status_code=404, detail=f"unknown run {name}")
    run_dir = (root / name).resolve()
    if not run_dir.is_relative_to(root.resolve()) or not (run_dir / RUN_FILE).is_file():
        raise HTTPException(status_code=404, detail=f"unknown run {name}")
    try:
        return _read_run(run_dir / RUN_FILE)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=f"run {name} has no readable run.json") from exc


def _read_run(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return data
