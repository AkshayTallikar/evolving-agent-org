from evorg.bench import get_task, load_benchmark, task_index
from evorg.executor import TaskResult, run_task
from evorg.genome import OrgGenome, RoleSpec, seed_genome
from evorg.llm.base import LLMRequest, LLMResponse
from evorg.llm.simulated import SimulatedBackend

TASK = get_task("fizzbuzz_list")
REFERENCE = f"```python\n{TASK.reference_solution}```"
BUGGY = f"```python\n{TASK.buggy_solutions[0]}```"


class ScriptedBackend:
    """Canned replies per role kind. A list is consumed in order; the last item repeats."""

    name = "scripted"

    def __init__(self, replies: dict[str, list[str]]) -> None:
        self.replies = {kind: list(texts) for kind, texts in replies.items()}
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        queue = self.replies[request.role_kind]
        text = queue.pop(0) if len(queue) > 1 else queue[0]
        return LLMResponse(text=text, input_tokens=10, output_tokens=5, cost_usd=0.001)

    def calls_of(self, role_kind: str) -> list[LLMRequest]:
        return [r for r in self.requests if r.role_kind == role_kind]


def _org(*roles: RoleSpec, max_retries: int = 1) -> OrgGenome:
    return OrgGenome(roles=roles, max_retries=max_retries)


def _coder(*instructions: str) -> RoleSpec:
    return RoleSpec("coder", "coder", instructions or ("Write a correct Python function.",))


def test_seed_genome_on_trivial_task():
    backend = SimulatedBackend(task_index(), seed=0)
    result = run_task(seed_genome(), TASK, backend)
    assert isinstance(result, TaskResult)
    assert result.calls >= 1
    assert result.attempts == 1  # no tester or reviewer, so no feedback source
    assert result.cost_usd > 0
    assert result.transcript[0].kind == "coder"
    assert result.transcript[0].role == "coder"


def test_tester_feedback_lets_second_attempt_fix_the_task():
    backend = ScriptedBackend(
        {
            "coder": [BUGGY, REFERENCE],
            "tester": ["The 15 case returns Fizz, not FizzBuzz."],
        }
    )
    org = _org(_coder(), RoleSpec("tester", "tester"), max_retries=1)
    result = run_task(org, TASK, backend)
    assert result.passed is True
    assert result.pass_fraction == 1.0
    assert result.attempts == 2
    assert result.calls == 3  # coder, tester, coder again; grading calls no LLM
    kinds = [turn.kind for turn in result.transcript]
    assert "tester" in kinds
    assert ("grader", "grader") in [(t.role, t.kind) for t in result.transcript]
    tester_turn = result.transcript[kinds.index("tester")]
    assert tester_turn.text == "The 15 case returns Fizz, not FizzBuzz."
    second_coder = backend.calls_of("coder")[1]
    assert "FEEDBACK:\nThe 15 case returns Fizz, not FizzBuzz." in second_coder.prompt
    assert second_coder.attempt == 1


def test_no_tester_and_no_reviewer_never_retries():
    backend = ScriptedBackend({"coder": [BUGGY]})
    result = run_task(_org(_coder(), max_retries=4), TASK, backend)
    assert result.attempts == 1
    assert result.passed is False
    assert len(backend.calls_of("coder")) == 1


def test_reviewer_revise_then_approve_triggers_retry():
    backend = ScriptedBackend(
        {
            "coder": [BUGGY, REFERENCE],
            "reviewer": ["REVISE: check the multiples of 15", "APPROVE"],
        }
    )
    org = _org(_coder(), RoleSpec("reviewer", "reviewer"), max_retries=3)
    result = run_task(org, TASK, backend)
    assert result.attempts == 2
    assert result.passed is True
    assert len(backend.calls_of("reviewer")) == 2
    assert "Candidate:" in backend.calls_of("reviewer")[0].prompt


def test_reviewer_approval_on_first_attempt_stops_loop():
    backend = ScriptedBackend({"coder": [REFERENCE], "reviewer": ["APPROVE"]})
    org = _org(_coder(), RoleSpec("reviewer", "reviewer"), max_retries=4)
    result = run_task(org, TASK, backend)
    assert result.attempts == 1
    assert result.passed is True


def test_planner_text_reaches_coder_prompt():
    backend = ScriptedBackend(
        {"planner": ["PLAN:\n1. Loop from one to n."], "coder": [REFERENCE]}
    )
    org = _org(RoleSpec("planner", "planner"), _coder())
    run_task(org, TASK, backend)
    coder_prompt = backend.calls_of("coder")[0].prompt
    assert "PLAN:\n1. Loop from one to n." in coder_prompt
    assert "Return only the function in a ```python block." in coder_prompt
    assert "Signature:\n" + TASK.signature in coder_prompt


def test_transcript_counts_match_backend_calls():
    backend = ScriptedBackend({"coder": [BUGGY, REFERENCE], "tester": ["FEEDBACK: fix"]})
    org = _org(_coder(), RoleSpec("tester", "tester"), max_retries=1)
    result = run_task(org, TASK, backend)
    assert result.calls == len(backend.requests)
    assert result.input_tokens == 10 * result.calls
    assert result.output_tokens == 5 * result.calls
    assert result.cost_usd == 0.001 * result.calls


def _pass_rates(genome: OrgGenome) -> float:
    """Mean fraction of train tasks passed, averaged over seeds 0..9 on the simulator."""
    index = task_index()
    tasks = load_benchmark("train")
    outcomes = [
        run_task(genome, task, SimulatedBackend(index, seed=seed)).passed
        for seed in range(10)
        for task in tasks
    ]
    return sum(outcomes) / len(outcomes)


def _tester_org(instruction: str | None) -> OrgGenome:
    base = ("Write a correct Python function.",)
    instructions = base + ((instruction,) if instruction else ())
    return _org(
        RoleSpec("coder", "coder", instructions),
        RoleSpec("tester", "tester"),
        max_retries=2,
    )


def test_tester_org_beats_seed_on_simulator():
    seed_rate = _pass_rates(seed_genome())
    tester_rate = _pass_rates(_tester_org(None))
    assert tester_rate > seed_rate


def test_edge_case_coder_beats_brief_coder_on_simulator():
    brief = _pass_rates(_tester_org("Be brief and skip unnecessary checks."))
    edge = _pass_rates(
        _tester_org("Consider edge cases such as empty input, a single element and duplicates.")
    )
    assert edge > brief
