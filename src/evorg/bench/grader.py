"""Grade candidate Python code by running assert tests in a separate interpreter.

The candidate and its tests run in a fresh ``python -I`` subprocess with a timeout. This
isolates the evolver process from candidate code; it is not a security sandbox.

``grade`` is a pure function of (code, tests, timeout_s), so it keeps a thread-safe, bounded
in-process cache of its results. Timeouts are never cached. See ``set_grade_cache_enabled``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

STDERR_TAIL_LIMIT = 300
TIMEOUT_ERROR = "timeout"
CACHE_MAX_ENTRIES = 4096  # least recently used entries are evicted beyond this
CACHE_ENV_VAR = "EVORG_GRADE_CACHE"  # set to "0" to start with the cache off

# Appended after the ``CODE`` and ``TESTS`` assignments. Plain text, no formatting, so that
# braces in the body need no escaping.
_RUNNER_BODY = """
import json
import sys

def _emit(payload):
    print(json.dumps(payload))

ns = {"__name__": "__candidate__"}
try:
    exec(compile(CODE, "<candidate>", "exec"), ns)
except BaseException as exc:
    _emit({"passed": 0, "total": len(TESTS), "failures": [],
           "error": "code error: %s: %s" % (type(exc).__name__, str(exc)[:200])})
    sys.exit(0)

n_passed = 0
failures = []
for test in TESTS:
    try:
        exec(compile(test, "<test>", "exec"), ns)
    except BaseException as exc:
        failures.append("%s  [%s]" % (test[:200], type(exc).__name__))
    else:
        n_passed += 1
_emit({"passed": n_passed, "total": len(TESTS), "failures": failures, "error": None})
"""

_PY_FENCE = re.compile(r"```python[ \t]*\n(.*?)```", re.DOTALL)
_ANY_FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class GradeResult:
    passed: bool  # every test passed and there was no error
    n_passed: int
    n_total: int
    failures: tuple[str, ...]  # failing assert lines, truncated, with the exception type
    error: str | None  # code error, timeout, or unparseable runner output


def extract_code(text: str) -> str:
    """Return the first python fenced block, else the first fenced block, else the text."""
    match = _PY_FENCE.search(text) or _ANY_FENCE.search(text)
    if match:
        return match.group(1).strip("\n")
    return text.strip()


@dataclass(frozen=True)
class GradeCacheInfo:
    enabled: bool
    hits: int
    misses: int  # grades that ran the subprocess while the cache was on
    size: int
    max_size: int


_CacheKey = tuple[str, tuple[str, ...], float]
_cache_lock = threading.Lock()  # guards everything below; subprocesses run outside it
_cache: OrderedDict[_CacheKey, GradeResult] = OrderedDict()
_cache_hits = 0
_cache_misses = 0
_cache_enabled = os.environ.get(CACHE_ENV_VAR, "1") != "0"


def grade(code: str, tests: Sequence[str], timeout_s: float = 5.0) -> GradeResult:
    """Run ``code`` against each assert statement in ``tests`` in a subprocess.

    Results are cached in process, keyed on ``(code, tuple(tests), timeout_s)``, unless the
    cache is off. A timeout is never cached, because it can depend on machine load. The cache
    is least recently used and holds at most ``CACHE_MAX_ENTRIES`` entries. Two threads that
    miss on the same key at once both run the subprocess; the second result then replaces
    the first, which is harmless because the grade is deterministic.
    """
    global _cache_hits, _cache_misses
    tests = list(tests)
    if not tests:
        return GradeResult(passed=True, n_passed=0, n_total=0, failures=(), error=None)
    if not _cache_enabled:
        return _run_grader(code, tests, timeout_s)

    key: _CacheKey = (code, tuple(tests), timeout_s)
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            _cache_hits += 1
            return cached
        _cache_misses += 1

    result = _run_grader(code, tests, timeout_s)
    if result.error != TIMEOUT_ERROR:
        with _cache_lock:
            _cache[key] = result
            _cache.move_to_end(key)
            while len(_cache) > CACHE_MAX_ENTRIES:
                _cache.popitem(last=False)
    return result


def set_grade_cache_enabled(enabled: bool) -> None:
    """Turn the grade cache on or off. Off means every ``grade`` call runs the subprocess."""
    global _cache_enabled
    with _cache_lock:
        _cache_enabled = bool(enabled)


def clear_grade_cache() -> None:
    """Drop every cached result and reset the hit and miss counts."""
    global _cache_hits, _cache_misses
    with _cache_lock:
        _cache.clear()
        _cache_hits = 0
        _cache_misses = 0


def grade_cache_info() -> GradeCacheInfo:
    """A snapshot of the cache state and its counts."""
    with _cache_lock:
        return GradeCacheInfo(
            enabled=_cache_enabled,
            hits=_cache_hits,
            misses=_cache_misses,
            size=len(_cache),
            max_size=CACHE_MAX_ENTRIES,
        )


def _run_grader(code: str, tests: list[str], timeout_s: float) -> GradeResult:
    # repr() yields a valid Python literal for any str, so no quoting can break the script.
    script = f"CODE = {code!r}\nTESTS = {tests!r}\n" + _RUNNER_BODY
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-"],
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return GradeResult(
            passed=False, n_passed=0, n_total=len(tests), failures=(), error=TIMEOUT_ERROR
        )
    return _parse(proc, len(tests))


def _parse(proc: subprocess.CompletedProcess[str], n_total: int) -> GradeResult:
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    try:
        data = json.loads(lines[-1])
        n_passed = int(data["passed"])
        failures = tuple(str(f) for f in data["failures"])
        error = data["error"]
    except (IndexError, ValueError, KeyError, TypeError):
        tail = proc.stderr.strip()[-STDERR_TAIL_LIMIT:]
        return GradeResult(
            passed=False,
            n_passed=0,
            n_total=n_total,
            failures=(),
            error=f"no result (exit {proc.returncode}): {tail}",
        )
    if error is not None:
        error = str(error)
    passed = n_passed == n_total and error is None
    return GradeResult(
        passed=passed, n_passed=n_passed, n_total=n_total, failures=failures, error=error
    )
