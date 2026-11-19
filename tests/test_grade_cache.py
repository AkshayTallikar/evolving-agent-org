"""The in-process grade cache: hit and miss counts, the off switch, timeouts, LRU, threads."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from evorg.bench import grader
from evorg.bench.grader import (
    CACHE_ENV_VAR,
    clear_grade_cache,
    grade,
    grade_cache_info,
    set_grade_cache_enabled,
)
from evorg.evolution import EvolutionConfig, run_simulated

ADD_TESTS = ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"]
GOOD_ADD = "def add(a, b):\n    return a + b\n"
BUGGY_ADD = "def add(a, b):\n    return a * b\n"
SYNTAX_ERROR = "def add(a, b)\n    return a"
LOOP_ADD = "def add(a, b):\n    while True:\n        pass\n"


@pytest.fixture(autouse=True)
def fresh_cache():
    """Every test starts with an empty, enabled cache and leaves it as it found it."""
    previous = grade_cache_info().enabled
    clear_grade_cache()
    set_grade_cache_enabled(True)
    yield
    clear_grade_cache()
    set_grade_cache_enabled(previous)


@pytest.fixture
def subprocess_calls(monkeypatch):
    """Count how many grader subprocesses start, still running them for real."""
    calls = []
    real_run = grader.subprocess.run

    def counting_run(*args, **kwargs):
        calls.append(1)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(grader.subprocess, "run", counting_run)
    return calls


def test_repeat_grade_is_a_hit_and_returns_the_same_result():
    first = grade(GOOD_ADD, ADD_TESTS)
    second = grade(GOOD_ADD, ADD_TESTS)

    assert first == second
    info = grade_cache_info()
    assert (info.hits, info.misses, info.size) == (1, 1, 1)
    assert info.enabled


def test_key_covers_code_tests_order_and_timeout(subprocess_calls):
    grade(GOOD_ADD, ADD_TESTS)
    grade(BUGGY_ADD, ADD_TESTS)  # different code
    grade(GOOD_ADD, ADD_TESTS[:2])  # different tests
    grade(GOOD_ADD, list(reversed(ADD_TESTS)))  # same tests, other order
    grade(GOOD_ADD, ADD_TESTS, timeout_s=4.0)  # other timeout

    info = grade_cache_info()
    assert (info.hits, info.misses, info.size) == (0, 5, 5)
    assert len(subprocess_calls) == 5


def test_equal_test_lists_share_an_entry(subprocess_calls):
    grade(GOOD_ADD, tuple(ADD_TESTS))
    grade(GOOD_ADD, list(ADD_TESTS))

    assert grade_cache_info().hits == 1
    assert len(subprocess_calls) == 1


def test_empty_tests_do_not_touch_the_cache():
    grade("this is not python (((", [])

    info = grade_cache_info()
    assert (info.hits, info.misses, info.size) == (0, 0, 0)


def test_disabled_cache_runs_every_call(subprocess_calls):
    set_grade_cache_enabled(False)

    results = [grade(GOOD_ADD, ADD_TESTS) for _ in range(3)]

    assert len(subprocess_calls) == 3
    assert all(r == results[0] for r in results)
    info = grade_cache_info()
    assert not info.enabled
    assert (info.hits, info.misses, info.size) == (0, 0, 0)


def test_cache_can_be_turned_back_on(subprocess_calls):
    set_grade_cache_enabled(False)
    grade(GOOD_ADD, ADD_TESTS)
    set_grade_cache_enabled(True)
    grade(GOOD_ADD, ADD_TESTS)
    grade(GOOD_ADD, ADD_TESTS)

    assert len(subprocess_calls) == 2
    assert (grade_cache_info().hits, grade_cache_info().misses) == (1, 1)


def test_env_var_zero_starts_with_cache_off():
    env = {**os.environ, CACHE_ENV_VAR: "0"}
    script = "from evorg.bench.grader import grade_cache_info as i; print(i().enabled)"
    out = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_timeouts_are_not_cached(subprocess_calls):
    first = grade(LOOP_ADD, ADD_TESTS, timeout_s=0.5)
    second = grade(LOOP_ADD, ADD_TESTS, timeout_s=0.5)

    assert first.error == "timeout"
    assert second == first
    info = grade_cache_info()
    assert (info.hits, info.misses, info.size) == (0, 2, 0)
    assert len(subprocess_calls) == 2


def test_other_results_are_cached_even_when_they_fail(subprocess_calls):
    grade(BUGGY_ADD, ADD_TESTS)
    grade(SYNTAX_ERROR, ADD_TESTS)
    grade(BUGGY_ADD, ADD_TESTS)
    grade(SYNTAX_ERROR, ADD_TESTS)

    assert len(subprocess_calls) == 2
    assert (grade_cache_info().hits, grade_cache_info().size) == (2, 2)


def test_least_recently_used_entry_is_evicted(monkeypatch, subprocess_calls):
    monkeypatch.setattr(grader, "CACHE_MAX_ENTRIES", 2)
    a, b, c = (f"def add(a, b):\n    return a + b + {k}\n" for k in range(3))
    tests = ["assert True"]

    grade(a, tests)
    grade(b, tests)
    grade(a, tests)  # hit: a becomes most recently used
    grade(c, tests)  # evicts b, the least recently used
    assert grade_cache_info().size == 2

    grade(a, tests)  # still cached
    grade(b, tests)  # evicted, so it runs again

    info = grade_cache_info()
    assert info.size == 2
    assert info.max_size == 2
    assert (info.hits, info.misses) == (2, 4)
    assert len(subprocess_calls) == 4


def test_concurrent_grades_are_consistent():
    codes = [GOOD_ADD, BUGGY_ADD, SYNTAX_ERROR]
    expected = {code: grader._run_grader(code, ADD_TESTS, 5.0) for code in codes}
    jobs = codes * 20

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda code: grade(code, ADD_TESTS), jobs))

    assert results == [expected[code] for code in jobs]
    info = grade_cache_info()
    assert info.hits + info.misses == len(jobs)
    assert info.size == len(codes)


def test_simulated_run_is_identical_with_cache_on_and_off():
    config = EvolutionConfig(population=4, generations=2, seed=0)

    set_grade_cache_enabled(True)
    clear_grade_cache()
    with_cache = json.dumps(run_simulated(config).to_dict(), sort_keys=True)
    hits = grade_cache_info().hits

    set_grade_cache_enabled(False)
    clear_grade_cache()
    without_cache = json.dumps(run_simulated(config).to_dict(), sort_keys=True)

    assert hits > 0, "the simulated run should repeat some grades"
    assert with_cache == without_cache
