import re

import pytest

from evorg.bench import get_task, grade, load_benchmark, task_index
from evorg.bench.tasks import Task

ALL_TASKS = load_benchmark("all")
TASK_IDS = [t.id for t in ALL_TASKS]


def _all_tests(task: Task) -> list[str]:
    return list(task.visible_tests) + list(task.hidden_tests)


def test_split_sizes():
    assert len(load_benchmark("train")) == 8
    assert len(load_benchmark("holdout")) == 4
    assert len(ALL_TASKS) == 12


def test_splits_are_labelled_and_ordered():
    assert all(t.split == "train" for t in load_benchmark("train"))
    assert all(t.split == "holdout" for t in load_benchmark("holdout"))
    assert ALL_TASKS == load_benchmark("train") + load_benchmark("holdout")


def test_unknown_split_raises_value_error():
    with pytest.raises(ValueError):
        load_benchmark("validation")


def test_get_task_unknown_id_raises_key_error():
    with pytest.raises(KeyError):
        get_task("no_such_task")


def test_task_index_covers_every_task():
    index = task_index()
    assert set(index) == set(TASK_IDS)
    assert index["fizzbuzz_list"] is get_task("fizzbuzz_list")


def test_ids_are_unique():
    assert len(set(TASK_IDS)) == len(TASK_IDS)


@pytest.mark.parametrize("task", ALL_TASKS, ids=TASK_IDS)
def test_difficulty_in_range(task):
    assert 0.0 <= task.difficulty <= 3.0


@pytest.mark.parametrize("task", ALL_TASKS, ids=TASK_IDS)
def test_test_counts(task):
    assert 2 <= len(task.visible_tests) <= 4
    assert 3 <= len(task.hidden_tests) <= 6


@pytest.mark.parametrize("task", ALL_TASKS, ids=TASK_IDS)
def test_tests_are_single_line_asserts_calling_the_function(task):
    name = re.match(r"def (\w+)\(", task.signature).group(1)
    for test in _all_tests(task):
        assert test.startswith("assert "), test
        assert "\n" not in test, test
        assert name in test, test


@pytest.mark.parametrize("task", ALL_TASKS, ids=TASK_IDS)
def test_reference_passes_all_tests(task):
    result = grade(task.reference_solution, _all_tests(task))
    assert result.passed, result


@pytest.mark.parametrize("task", ALL_TASKS, ids=TASK_IDS)
def test_every_buggy_solution_fails_some_test(task):
    assert len(task.buggy_solutions) >= 2
    for bug in task.buggy_solutions:
        assert not grade(bug, _all_tests(task)).passed, bug


@pytest.mark.parametrize("task", ALL_TASKS, ids=TASK_IDS)
def test_some_buggy_solution_fails_a_visible_test(task):
    assert any(not grade(bug, task.visible_tests).passed for bug in task.buggy_solutions)


@pytest.mark.parametrize("task", ALL_TASKS, ids=TASK_IDS)
def test_some_buggy_solution_is_subtle(task):
    """Passes every visible test but fails at least one hidden test."""
    assert any(
        grade(bug, task.visible_tests).passed and not grade(bug, task.hidden_tests).passed
        for bug in task.buggy_solutions
    )
