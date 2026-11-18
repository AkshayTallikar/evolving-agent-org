import json
from types import SimpleNamespace

import pytest

from evorg.genes import INSTRUCTION_POOL, MODEL_TIERS, ROLE_KINDS
from evorg.llm.base import LLMRequest
from evorg.llm.simulated import FEATURE_WEIGHTS, SimulatedBackend, skill

REFERENCE = "def add_one(x: int) -> int:\n    return x + 1"
BUGGY = ("def add_one(x: int) -> int:\n    return x + 2",)
TASK = SimpleNamespace(reference_solution=REFERENCE, buggy_solutions=BUGGY, difficulty=1.0)

EDGE_SYSTEM = "You write Python.\n- Consider edge cases such as empty input and duplicates."
BRIEF_SYSTEM = "You write Python.\n- Be brief and skip unnecessary checks."


def _backend(seed: int = 0) -> SimulatedBackend:
    return SimulatedBackend({"t1": TASK}, seed=seed)


def _coder_request(system: str, i: int, model: str = "small") -> LLMRequest:
    return LLMRequest(
        "coder", model, system, f"Implement add_one, variant {i}.", task_id="t1", attempt=0
    )


def _code_of(text: str) -> str:
    return text.split("```python\n", 1)[1].rsplit("\n```", 1)[0]


def test_backend_name_is_sim():
    assert SimulatedBackend({}).name == "sim"


# --- keyword model -------------------------------------------------------------------


def test_every_pool_instruction_matches_its_intended_keyword():
    expected = {
        INSTRUCTION_POOL[0]: "edge case",
        INSTRUCTION_POOL[1]: "step by step",
        INSTRUCTION_POOL[2]: "mentally test",
        INSTRUCTION_POOL[3]: "off-by-one",
        INSTRUCTION_POOL[4]: "empty input",
        INSTRUCTION_POOL[5]: "type hint",
        INSTRUCTION_POOL[6]: "docstring",
        INSTRUCTION_POOL[7]: "descriptive",
        INSTRUCTION_POOL[9]: "be brief",
        INSTRUCTION_POOL[10]: "guess",
        INSTRUCTION_POOL[11]: "speed over correctness",
    }
    for text, keyword in expected.items():
        assert keyword in text.lower(), (keyword, text)
    # "skip" is the second harmful keyword of the "be brief" instruction.
    assert "skip" in INSTRUCTION_POOL[9].lower()
    # The neutral line about function length carries no keyword at all.
    neutral = INSTRUCTION_POOL[8].lower()
    assert not [kw for kw in FEATURE_WEIGHTS if kw in neutral]


def test_keyword_counts_once_however_often_it_appears():
    once = LLMRequest("coder", "small", "edge case", "", task_id="t1")
    thrice = LLMRequest("coder", "small", "edge case edge case edge case", "", task_id="t1")
    assert skill(once) == pytest.approx(FEATURE_WEIGHTS["edge case"])
    assert skill(thrice) == pytest.approx(FEATURE_WEIGHTS["edge case"])


def test_large_model_and_plan_and_feedback_bonuses():
    small = LLMRequest("coder", "small", "x", "y")
    large = LLMRequest("coder", "large", "x", "y")
    assert skill(large) - skill(small) == pytest.approx(0.8)
    planned = LLMRequest("coder", "small", "x", "PLAN:\n1. do it")
    assert skill(planned) - skill(small) == pytest.approx(0.4)
    feedback = LLMRequest("coder", "small", "x", "FEEDBACK: fix it")
    assert skill(feedback) - skill(small) == pytest.approx(0.6)


def test_instruction_lines_beyond_six_are_penalised():
    lines = "\n".join(f"- rule {i}" for i in range(9))
    base = LLMRequest("coder", "small", "x", "y")
    crowded = LLMRequest("coder", "small", lines, "y")
    assert skill(crowded) == pytest.approx(skill(base) - 3 * 0.05)


# --- coder ---------------------------------------------------------------------------


def test_coder_returns_fenced_python_block():
    text = _backend().complete(_coder_request(EDGE_SYSTEM, 0)).text
    assert text.startswith("```python\n") and text.endswith("\n```")


def test_coder_is_deterministic_for_identical_requests():
    request = _coder_request(EDGE_SYSTEM, 3)
    assert _backend().complete(request) == _backend().complete(request)


def test_different_seed_can_change_the_outcome():
    outputs = {
        (seed, i): _code_of(_backend(seed).complete(_coder_request(BRIEF_SYSTEM, i)).text)
        for seed in (0, 1)
        for i in range(60)
    }
    assert any(outputs[(0, i)] != outputs[(1, i)] for i in range(60))


def test_coder_success_rate_higher_with_edge_case_prompt():
    backend = _backend()
    n = 300

    def rate(system: str, model: str = "small") -> float:
        hits = sum(
            _code_of(backend.complete(_coder_request(system, i, model)).text) == REFERENCE
            for i in range(n)
        )
        return hits / n

    edge = rate(EDGE_SYSTEM)
    brief = rate(BRIEF_SYSTEM)
    assert edge > brief + 0.2
    assert rate(BRIEF_SYSTEM, model="large") > brief + 0.1


def test_coder_failure_returns_a_buggy_solution():
    backend = _backend()
    codes = {
        _code_of(backend.complete(_coder_request(BRIEF_SYSTEM, i)).text) for i in range(50)
    }
    assert codes <= {REFERENCE, *BUGGY}
    assert set(BUGGY) & codes


def test_coder_unknown_task_raises_key_error():
    request = LLMRequest("coder", "small", EDGE_SYSTEM, "p", task_id="nope")
    with pytest.raises(KeyError):
        _backend().complete(request)


# --- planner -------------------------------------------------------------------------


def test_planner_emits_plan_and_propagates_positive_keywords():
    system = "Plan carefully.\n- Think about edge cases.\n- Work step by step.\n- Be brief."
    text = _backend().complete(LLMRequest("planner", "small", system, "Task?")).text
    assert text.startswith("PLAN:\n1. ")
    focus = [line for line in text.splitlines() if line.startswith("Focus: ")]
    assert focus == ["Focus: edge case, step by step"]


def test_planner_without_keywords_has_no_focus_line():
    text = _backend().complete(LLMRequest("planner", "small", "Plan.", "Task?")).text
    assert "Focus:" not in text


def test_planner_plan_reaches_the_coder_as_a_bonus():
    plan = _backend().complete(LLMRequest("planner", "small", EDGE_SYSTEM, "Task?")).text
    with_plan = LLMRequest("coder", "small", BRIEF_SYSTEM, f"Task.\n{plan}", task_id="t1")
    without = LLMRequest("coder", "small", BRIEF_SYSTEM, "Task.", task_id="t1")
    assert skill(with_plan) > skill(without) + 0.4 - 1e-9


# --- tester --------------------------------------------------------------------------


def test_tester_echoes_failing_asserts():
    prompt = (
        "Visible tests run:\n"
        "assert add_one(1) == 2\n"
        "some noise line\n"
        "assert add_one(0) == 1"
    )
    text = _backend().complete(LLMRequest("tester", "small", "Test.", prompt)).text
    assert text.startswith("FEEDBACK: ")
    assert "assert add_one(1) == 2" in text
    assert "assert add_one(0) == 1" in text
    assert "noise" not in text


def test_tester_without_failing_lines_says_tests_failed():
    text = _backend().complete(LLMRequest("tester", "small", "Test.", "nothing")).text
    assert text == "FEEDBACK: tests failed"


# --- reviewer ------------------------------------------------------------------------


def _review_request(code: str, i: int) -> LLMRequest:
    prompt = f"Review attempt {i}:\n```python\n{code}\n```"
    return LLMRequest("reviewer", "small", "Review code carefully.", prompt, task_id="t1")


def test_reviewer_catches_buggy_code_more_often_than_it_flags_correct_code():
    backend = _backend()
    n = 300
    caught = sum(
        backend.complete(_review_request(BUGGY[0], i)).text.startswith("REVISE")
        for i in range(n)
    )
    false_alarms = sum(
        backend.complete(_review_request(REFERENCE, i)).text.startswith("REVISE")
        for i in range(n)
    )
    assert caught > false_alarms + 0.2 * n
    assert false_alarms < 0.15 * n


def test_reviewer_outputs_only_known_verdicts():
    backend = _backend()
    for i in range(20):
        text = backend.complete(_review_request(BUGGY[0], i)).text
        assert text == "APPROVE" or text.startswith("REVISE: ")


def test_reviewer_unknown_task_raises_key_error():
    request = LLMRequest("reviewer", "small", "r", "```python\nx\n```", task_id="nope")
    with pytest.raises(KeyError):
        _backend().complete(request)


# --- architect -----------------------------------------------------------------------

SAMPLE_GENOME = {
    "roles": [
        {
            "name": "planner",
            "kind": "planner",
            "instructions": [INSTRUCTION_POOL[0]],
            "model": "small",
        },
        {
            "name": "coder",
            "kind": "coder",
            "instructions": ["Write a correct Python function."],
            "model": "small",
        },
        {"name": "reviewer", "kind": "reviewer", "instructions": [], "model": "large"},
    ],
    "max_retries": 2,
    "parent_ids": [],
    "generation": 0,
    "mutation_note": "",
}


def _architect(genome: dict, n: int, seed: int = 0) -> list[dict]:
    request = LLMRequest(
        "architect", "large", "Propose mutations.", "genome...", metadata={"genome": genome, "n": n}
    )
    response = _backend(seed).complete(request)
    return json.loads(response.text)["mutations"]


def _apply_all(genome: dict, mutations: list[dict]) -> dict:
    """Apply mutations in order and check each against the genome rules (design section 4.2)."""
    roles = [dict(r, instructions=list(r["instructions"])) for r in genome["roles"]]
    max_retries = genome["max_retries"]
    for m in mutations:
        assert set(m) == {"op", "role", "value", "index", "rationale"}
        assert m["rationale"]
        op = m["op"]
        by_name = {r["name"]: r for r in roles}
        if op == "add_instruction":
            role = by_name[m["role"]]
            assert len(role["instructions"]) < 8
            assert m["value"] in INSTRUCTION_POOL and m["value"] not in role["instructions"]
            role["instructions"].append(m["value"])
        elif op == "replace_instruction":
            role = by_name[m["role"]]
            assert 0 <= m["index"] < len(role["instructions"])
            assert m["value"] in INSTRUCTION_POOL and m["value"] not in role["instructions"]
            role["instructions"][m["index"]] = m["value"]
        elif op == "remove_instruction":
            role = by_name[m["role"]]
            assert 0 <= m["index"] < len(role["instructions"])
            del role["instructions"][m["index"]]
        elif op == "add_role":
            assert m["value"] in ("planner", "tester", "reviewer")
            assert len(roles) < 6 and m["role"] not in by_name
            assert m["role"] in (m["value"], m["value"] + "2")
            kinds = [r["kind"] for r in roles]
            assert kinds.count(m["value"]) < {"planner": 2, "tester": 1, "reviewer": 1}[m["value"]]
            roles.append({"name": m["role"], "kind": m["value"], "instructions": [],
                          "model": "small"})
        elif op == "remove_role":
            role = by_name[m["role"]]
            assert role["kind"] != "coder"
            roles.remove(role)
        elif op == "change_model":
            role = by_name[m["role"]]
            assert m["value"] in MODEL_TIERS and m["value"] != role["model"]
            role["model"] = m["value"]
        elif op == "set_retries":
            assert 0 <= m["value"] <= 4 and m["value"] != max_retries
            max_retries = m["value"]
        else:
            raise AssertionError(f"unknown op {op}")
    # Final organization is valid.
    assert 1 <= len(roles) <= 6
    assert len({r["name"] for r in roles}) == len(roles)
    assert all(r["kind"] in ROLE_KINDS for r in roles)
    assert [r["kind"] for r in roles].count("coder") == 1
    assert all(len(r["instructions"]) <= 8 for r in roles)
    assert all(len(set(r["instructions"])) == len(r["instructions"]) for r in roles)
    assert 0 <= max_retries <= 4
    return {"roles": roles, "max_retries": max_retries}


@pytest.mark.parametrize("n", [1, 3, 5])
def test_architect_returns_exactly_n_valid_mutations_across_seeds(n):
    kinds = set()
    for seed in range(40):
        mutations = _architect(SAMPLE_GENOME, n, seed=seed)
        assert len(mutations) == n
        _apply_all(SAMPLE_GENOME, mutations)
        kinds.update(m["op"] for m in mutations)
    assert {"add_instruction", "set_retries", "change_model"} <= kinds


def test_architect_never_removes_coder_or_adds_second_coder():
    for seed in range(60):
        for m in _architect(SAMPLE_GENOME, 4, seed=seed):
            assert not (m["op"] == "remove_role" and m["role"] == "coder")
            assert not (m["op"] == "add_role" and m["value"] == "coder")


def test_architect_handles_minimal_genome_with_only_coder():
    seed_genome = {
        "roles": [
            {"name": "coder", "kind": "coder",
             "instructions": ["Write a correct Python function."], "model": "small"}
        ],
        "max_retries": 1,
    }
    for seed in range(30):
        mutations = _architect(seed_genome, 3, seed=seed)
        assert len(mutations) == 3
        _apply_all(seed_genome, mutations)


def test_architect_is_deterministic():
    assert _architect(SAMPLE_GENOME, 3, seed=5) == _architect(SAMPLE_GENOME, 3, seed=5)
