from evorg.bench.grader import GradeResult, extract_code, grade

ADD_TESTS = ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"]
GOOD_ADD = "def add(a, b):\n    return a + b\n"


def test_correct_code_passes_all_tests():
    result = grade(GOOD_ADD, ADD_TESTS)
    assert result == GradeResult(passed=True, n_passed=3, n_total=3, failures=(), error=None)


def test_partial_pass_reports_failing_assert_text():
    buggy = "def add(a, b):\n    return a * b\n"
    result = grade(buggy, ADD_TESTS)
    assert not result.passed
    assert result.n_passed == 1
    assert result.n_total == 3
    assert result.error is None
    assert len(result.failures) == 2
    assert result.failures[0].startswith("assert add(1, 2) == 3")
    assert "[AssertionError]" in result.failures[0]


def test_failure_text_is_truncated():
    long_test = "assert add(1, 2) == 4  # " + "x" * 400
    result = grade(GOOD_ADD, [long_test])
    assert result.n_passed == 0
    assert len(result.failures) == 1
    assert len(result.failures[0]) < 200 + len("  [AssertionError]") + 1


def test_syntax_error_in_code_is_reported():
    result = grade("def add(a, b)\n    return a", ADD_TESTS)
    assert not result.passed
    assert result.n_passed == 0
    assert result.n_total == 3
    assert result.failures == ()
    assert result.error is not None
    assert result.error.startswith("code error: SyntaxError")


def test_module_level_runtime_error_is_reported():
    result = grade("x = 1\nraise ValueError('boom')\n", ADD_TESTS)
    assert not result.passed
    assert result.n_passed == 0
    assert result.error == "code error: ValueError: boom"


def test_runtime_error_inside_function_fails_each_test():
    result = grade("def add(a, b):\n    return a / 0\n", ADD_TESTS)
    assert not result.passed
    assert result.n_passed == 0
    assert result.error is None
    assert any("[ZeroDivisionError]" in f for f in result.failures)


def test_infinite_loop_times_out():
    code = "def add(a, b):\n    while True:\n        pass\n"
    result = grade(code, ADD_TESTS, timeout_s=1.0)
    assert result == GradeResult(
        passed=False, n_passed=0, n_total=3, failures=(), error="timeout"
    )


def test_quotes_triple_quotes_and_backslashes_survive_embedding():
    code = (
        "def tag(s):\n"
        '    """Docstring with \\"quotes\\" and \'\'\' and \\\\ backslash."""\n'
        '    return "\\\\" + s + \'"\'\n'
    )
    tests = [
        "assert tag('x') == chr(92) + 'x' + chr(34)",
        "assert tag('') == chr(92) + chr(34)",
        "assert tag.__doc__.endswith('backslash.')",
    ]
    result = grade(code, tests)
    assert result.passed, result


def test_empty_test_list_passes_without_running():
    assert grade("this is not python (((", []) == GradeResult(
        passed=True, n_passed=0, n_total=0, failures=(), error=None
    )


def test_extract_code_prefers_python_block():
    text = "Here it is:\n```python\ndef f():\n    return 1\n```\nThanks."
    assert extract_code(text) == "def f():\n    return 1"


def test_extract_code_takes_first_python_block():
    text = "```python\nfirst = 1\n```\nand\n```python\nsecond = 2\n```"
    assert extract_code(text) == "first = 1"


def test_extract_code_python_block_beats_earlier_generic_block():
    text = "```\nold = 0\n```\n```python\nnew = 1\n```"
    assert extract_code(text) == "new = 1"


def test_extract_code_generic_fence():
    assert extract_code("```\nx = 1\n```") == "x = 1"


def test_extract_code_no_fence_returns_stripped_text():
    assert extract_code("  x = 2  \n") == "x = 2"
