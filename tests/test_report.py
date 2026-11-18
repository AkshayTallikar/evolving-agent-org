from dataclasses import replace

import pytest

from evorg.evolution import EvolutionConfig, run_simulated
from evorg.report import SIM_NOTE, render_markdown, sparkline

HEADINGS = [
    "# evorg run",
    "## Fitness by generation",
    "## Champion",
    "## What changed from the seed",
    "## Lineage",
    "## Held-out tasks",
]


@pytest.fixture(scope="module")
def tiny_run():
    return run_simulated(EvolutionConfig(population=3, generations=1, seed=0))


def test_sparkline_empty_is_empty():
    assert sparkline([]) == ""


def test_sparkline_equal_values_repeat_middle_block():
    assert sparkline([0.5, 0.5, 0.5]) == "▅▅▅"


def test_sparkline_scales_min_to_lowest_and_max_to_highest():
    line = sparkline([3.0, 1.0, 2.0])
    assert line[1] == "▁"
    assert line[0] == "█"
    assert len(line) == 3


def test_sparkline_is_monotonic_for_increasing_values():
    line = sparkline([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert line == "▁▂▃▄▅▆▇█"


def test_render_has_every_section_and_champion(tiny_run):
    text = render_markdown(tiny_run)
    for heading in HEADINGS:
        assert heading in text
    assert tiny_run.champion_id in text
    assert "Held-out tasks were never used for selection." in text
    assert text.rstrip().endswith(SIM_NOTE)


def test_render_shows_best_fitness_sparkline(tiny_run):
    text = render_markdown(tiny_run)
    best = [record.members[0][1] for record in tiny_run.generations]
    assert f"Best fitness: {sparkline(best)}" in text


def test_render_no_sim_note_for_other_backends(tiny_run):
    text = render_markdown(replace(tiny_run, backend="anthropic"))
    assert SIM_NOTE not in text


def test_render_says_so_when_champion_is_seed(tiny_run):
    text = render_markdown(replace(tiny_run, champion_id=tiny_run.seed_id))
    assert "No change: the seed was never beaten." in text


def test_render_header_mentions_simulator_seeds_only_when_several(tiny_run):
    single = render_markdown(tiny_run)
    assert "simulator seeds" not in single.split("## ")[0]

    multi = render_markdown(replace(tiny_run, config=replace(tiny_run.config, eval_seeds=3)))
    assert "scored over 3 simulator seeds" in multi.split("## ")[0]


def test_render_says_held_out_seeds_were_unseen_for_sim_only(tiny_run):
    note = "Held-out tasks were scored on simulator seeds that selection never saw."
    assert note in render_markdown(tiny_run)
    assert note not in render_markdown(replace(tiny_run, backend="anthropic"))
