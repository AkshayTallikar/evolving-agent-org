"""Run one organization on one task: planners, a coder, and a feedback-driven retry loop.

The loop follows design section 4.5. Planners write a plan once. The coder writes code and,
after the first attempt, sees feedback from whichever signal the organization has: a tester
that turns failing visible tests into feedback, or else a reviewer that may say REVISE. An
organization with neither has no signal, so its retries are never used.
"""

from __future__ import annotations

from dataclasses import dataclass

from evorg.bench import Task, extract_code, grade
from evorg.genome import OrgGenome, RoleSpec
from evorg.llm.base import LLMBackend, LLMRequest, LLMResponse

CODER_FORMAT_NOTE = "Return only the function in a ```python block."


@dataclass(frozen=True)
class Turn:
    role: str
    kind: str
    text: str


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    passed: bool
    pass_fraction: float  # over visible + hidden tests
    attempts: int
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    transcript: tuple[Turn, ...]


class _Run:
    """Mutable bookkeeping for one ``run_task`` call: call counts, tokens, transcript."""

    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.transcript: list[Turn] = []

    def call(self, role: RoleSpec, prompt: str, task: Task, attempt: int = 0) -> str:
        request = LLMRequest(
            role_kind=role.kind,
            model=role.model,
            system=role.system_prompt(),
            prompt=prompt,
            task_id=task.id,
            attempt=attempt,
        )
        response: LLMResponse = self.backend.complete(request)
        self.calls += 1
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        self.cost_usd += response.cost_usd
        self.transcript.append(Turn(role=role.name, kind=role.kind, text=response.text))
        return response.text


def _task_block(task: Task) -> str:
    tests = "\n".join(task.visible_tests)
    return (
        f"{task.title}\n\n{task.prompt}\n\n"
        f"Signature:\n{task.signature}\n\n"
        f"Examples (these tests must pass):\n{tests}"
    )


def _coder_prompt(task: Task, plan: str, feedback: str | None) -> str:
    parts = [_task_block(task)]
    if plan:
        parts.append(plan)
    if feedback:
        parts.append("FEEDBACK:\n" + feedback)
    parts.append(CODER_FORMAT_NOTE)
    return "\n\n".join(parts)


def _failing_lines(failures: tuple[str, ...], error: str | None) -> list[str]:
    lines = list(failures)
    if error:
        lines.append(error)
    return lines


def run_task(genome: OrgGenome, task: Task, backend: LLMBackend) -> TaskResult:
    """Run ``genome`` on ``task`` and grade the final code on visible + hidden tests."""
    run = _Run(backend)
    coder = genome.roles_of_kind("coder")[0]
    testers = genome.roles_of_kind("tester")
    reviewers = genome.roles_of_kind("reviewer")
    visible = list(task.visible_tests)

    plan = "\n\n".join(
        run.call(role, _task_block(task), task) for role in genome.roles_of_kind("planner")
    )

    attempts = 0
    code = ""
    feedback: str | None = None
    for attempt in range(1 + genome.max_retries):
        attempts += 1
        code = extract_code(
            run.call(coder, _coder_prompt(task, plan, feedback), task, attempt)
        )
        feedback = None

        if testers:
            graded = grade(code, visible)
            run.transcript.append(
                Turn(
                    role="grader",
                    kind="grader",
                    text=f"visible tests: {graded.n_passed}/{graded.n_total} passed",
                )
            )
            if not graded.passed:
                failing = _failing_lines(graded.failures, graded.error)
                prompt = (
                    f"The coder's code:\n```python\n{code}\n```\n"
                    "Failing tests:\n" + "\n".join(failing)
                )
                feedback = run.call(testers[0], prompt, task, attempt).strip() or None

        if feedback is None and reviewers:
            prompt = (
                f"{_task_block(task)}\n\n"
                f"Candidate:\n```python\n{code}\n```\n"
                "Reply APPROVE or REVISE: <reason>."
            )
            text = run.call(reviewers[0], prompt, task, attempt)
            if text.startswith("REVISE"):
                feedback = text.strip()

        if feedback is None:
            break

    final = grade(code, visible + list(task.hidden_tests))
    n_total = final.n_total
    pass_fraction = final.n_passed / n_total if n_total else 0.0
    return TaskResult(
        task_id=task.id,
        passed=final.passed,
        pass_fraction=pass_fraction,
        attempts=attempts,
        calls=run.calls,
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        cost_usd=run.cost_usd,
        transcript=tuple(run.transcript),
    )
