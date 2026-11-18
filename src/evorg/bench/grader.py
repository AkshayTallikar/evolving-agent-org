"""Grade candidate Python code by running assert tests in a separate interpreter.

The candidate and its tests run in a fresh ``python -I`` subprocess with a timeout. This
isolates the evolver process from candidate code; it is not a security sandbox.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass

STDERR_TAIL_LIMIT = 300

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


def grade(code: str, tests: Sequence[str], timeout_s: float = 5.0) -> GradeResult:
    """Run ``code`` against each assert statement in ``tests`` in a subprocess."""
    tests = list(tests)
    if not tests:
        return GradeResult(passed=True, n_passed=0, n_total=0, failures=(), error=None)

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
            passed=False, n_passed=0, n_total=len(tests), failures=(), error="timeout"
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
