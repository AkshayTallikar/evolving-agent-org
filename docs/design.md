# Design: self-evolving agent organizations

## 1. Problem

Multi-agent systems are usually designed by hand. Someone decides there is a planner, a
coder and a reviewer, writes their prompts, picks a model for each, and then tunes all of
it by intuition. The organization chart of an agent team is a design decision with large
effects on quality and cost, yet it is rarely searched systematically.

`evorg` treats the organization itself as the thing being optimised. An organization is a
**genome**: its roles, each role's instructions, the model tier each role runs on, and how
many feedback loops the team may spend. Organizations **propose changes to themselves**
through an architect agent, and a **tournament on a coding benchmark** decides which
variants survive. Fitness rewards correctness and penalises cost, so the search has to
discover when an extra reviewer or a larger model is worth paying for.

## 2. Goals for milestone 1

1. Run end to end on a small benchmark with no API key, deterministically, in CI.
2. Grade every candidate solution by actually executing it against unit tests.
3. Keep the LLM layer pluggable: the same loop runs on a simulated backend or on Claude.
4. Record full lineage, so every champion can be traced back to the seed one mutation at
   a time, with the architect's rationale for each step.
5. Report generalisation on held-out tasks, not only on the tasks evolution saw.

Non-goals for milestone 1: a web UI, crossover, parallel evaluation, Pareto selection.
Those are milestone 2.

## 3. Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │                  Evolver                     │
                 │  population ─► tournament ─► survivors       │
                 └───────┬──────────────────────────▲───────────┘
          parent genome  │                          │ fitness = pass_rate − λ·cost
          + eval report  ▼                          │
                 ┌───────────────┐          ┌───────┴────────┐
                 │   Architect   │          │   Evaluator    │
                 │ (large model) │          │ runs org on    │
                 │ proposes      │          │ every task     │
                 │ Mutations     │          └───────┬────────┘
                 └───────┬───────┘                  │
                         ▼ apply_mutation           ▼
                 ┌───────────────┐          ┌────────────────┐     ┌──────────┐
                 │  child genome │─────────►│    Executor    │────►│  Grader  │
                 └───────────────┘          │ planner→coder  │     │ subprocess│
                                            │ →tester/review │     │ + tests  │
                                            │ retry loop     │     └──────────┘
                                            └───────┬────────┘
                                                    ▼
                                            ┌────────────────┐
                                            │   LLMBackend   │  sim | anthropic
                                            └────────────────┘
```

## 4. Module contracts

All code lives in `src/evorg`. Every public function has type hints. Modules depend only on
modules listed above them.

### 4.1 `evorg.genes` (exists)

`INSTRUCTION_POOL`, `ROLE_KINDS = ("planner", "coder", "tester", "reviewer")`,
`MODEL_TIERS = ("small", "large")`.

### 4.2 `evorg.genome`

```python
class GenomeError(ValueError): ...

@dataclass(frozen=True)
class RoleSpec:
    name: str                       # unique within the org, e.g. "coder"
    kind: str                       # one of ROLE_KINDS
    instructions: tuple[str, ...] = ()
    model: str = "small"            # one of MODEL_TIERS
    def system_prompt(self) -> str  # fixed persona for the kind + "- " bullet per instruction

@dataclass(frozen=True)
class OrgGenome:
    roles: tuple[RoleSpec, ...]
    max_retries: int = 1            # 0..4 extra coder attempts driven by feedback
    parent_ids: tuple[str, ...] = ()
    generation: int = 0
    mutation_note: str = ""         # how this genome was created, human readable
    @property
    def genome_id(self) -> str      # first 10 hex chars of sha1 over canonical JSON of
                                    # roles + max_retries ONLY (lineage fields excluded)
    def role(self, name: str) -> RoleSpec            # KeyError if missing
    def roles_of_kind(self, kind: str) -> tuple[RoleSpec, ...]
    def validate(self) -> None      # raises GenomeError
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> "OrgGenome"

def seed_genome() -> OrgGenome      # one role: name "coder", kind "coder", model "small",
                                    # instructions ("Write a correct Python function.",),
                                    # max_retries 1
def diff_genomes(a: OrgGenome, b: OrgGenome) -> list[str]   # readable lines, e.g.
                                    # "+ role reviewer (reviewer, small)",
                                    # "coder: + instruction 'Handle empty input explicitly.'",
                                    # "coder: model small -> large", "max_retries 1 -> 2"
```

Validation rules: 1 to 6 roles; names unique and non-empty; every kind in `ROLE_KINDS`;
exactly one `coder`; at most 2 `planner`, at most 1 `tester`, at most 1 `reviewer`; every
model in `MODEL_TIERS`; at most 8 instructions per role, each 1 to 200 characters, no
duplicates within a role; `0 <= max_retries <= 4`.

### 4.3 `evorg.llm`

`evorg/llm/base.py`:

```python
MODEL_IDS = {"small": "claude-haiku-5-5", "large": "claude-opus-5-5"}
PRICING_PER_MTOK = {"small": (0.10, 0.50), "large": (4.00, 20.00)}  # (input, output) USD

@dataclass(frozen=True)
class LLMRequest:
    role_kind: str                  # a ROLE_KIND or "architect"
    model: str                      # tier: "small" | "large"
    system: str
    prompt: str
    task_id: str | None = None
    attempt: int = 0
    metadata: dict = field(default_factory=dict, hash=False, compare=False)

@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float

class LLMBackend(Protocol):
    name: str
    def complete(self, request: LLMRequest) -> LLMResponse: ...

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float
def count_tokens_approx(text: str) -> int       # max(1, len(text) // 4)
```

`evorg/llm/simulated.py`, `SimulatedBackend(task_index: Mapping[str, Any], seed: int = 0)`.
A deterministic stand-in for an LLM so the whole system runs offline. It is not a toy random
number generator: it encodes a simple, documented **capability model**, so that the
structure of an organization has measurable consequences that evolution can discover.
`task_index` maps task id to an object with the attributes of `bench.Task` (it only needs
`reference_solution`, `buggy_solutions`, `difficulty`).

All randomness comes from `_unit(*parts) -> float in [0,1)`, a sha256 of `seed` and the parts.
Same inputs, same output, on every machine.

Capability model (constants at module top, documented):

* `FEATURE_WEIGHTS`: lowercase keyword → weight. Helpful phrases (`"edge case"`,
  `"empty input"`, `"step by step"`, `"mentally test"`, `"off-by-one"`, `"type hint"`)
  have positive weight; harmful ones (`"be brief"`, `"skip"`, `"guess"`, `"speed over
  correctness"`) negative; neutral ones near zero. Each keyword counts at most once.
* `skill(request) = Σ weights of keywords found in system + prompt`
  `+ LARGE_MODEL_BONUS (0.8) if model == "large"`
  `+ PLAN_BONUS (0.4) if "PLAN:" in prompt`
  `+ FEEDBACK_BONUS (0.6) if "FEEDBACK:" in prompt`
  `− 0.05 per instruction line beyond 6` (instruction lines are lines starting with "- ").
* `p_success = sigmoid(skill + BASE_SKILL − difficulty)` with `BASE_SKILL = 0.5`.

Behaviour by `role_kind`:

* `coder`: with probability `p_success` return the task's reference solution, otherwise one
  of its buggy solutions (picked by `_unit`). Text is a fenced ```python block.
  Requires `task_id`; unknown id raises `KeyError`.
* `planner`: returns `"PLAN:\n1. ...\n2. ..."`. The plan repeats any FEATURE_WEIGHTS keywords
  found in the planner's own system prompt, so a planner told to think about edge cases
  hands that focus to the coder.
* `tester`: the executor runs the visible tests itself and puts the failures in the prompt;
  the tester's job is to turn them into actionable feedback. Returns
  `"FEEDBACK: " + summary` of the failing lines in the prompt.
* `reviewer`: reads the code in the prompt. If the code equals a buggy solution of the task,
  it catches it with probability `sigmoid(reviewer_skill − 0.5)` and returns
  `"REVISE: <hint>"`; otherwise `"APPROVE"`. A correct solution gets a false `REVISE` with
  probability 0.05.
* `architect`: `metadata["genome"]` holds `OrgGenome.to_dict()`. Returns a JSON object
  `{"mutations": [...]}` with `metadata.get("n", 1)` mutations, each valid for that genome,
  in the schema of §4.5. Ops are chosen with fixed weights; instructions come from
  `genes.INSTRUCTION_POOL`.

Token counts use `count_tokens_approx`; cost uses `estimate_cost`.

`evorg/llm/anthropic_backend.py`, `AnthropicBackend(client=None, effort="low")`. Imports
`anthropic` lazily inside `__init__` and raises a clear `RuntimeError` naming
`pip install evorg[anthropic]` if it is missing. Maps tier → `MODEL_IDS`, calls
`client.messages.create(model=..., max_tokens=4000, system=..., messages=[{"role": "user",
"content": prompt}], output_config={"effort": effort})`, joins the `text` blocks, reads
`usage.input_tokens/output_tokens`, raises `RuntimeError` if `stop_reason == "refusal"`.
Never constructed in tests unless a fake client is passed.

`evorg/llm/__init__.py`: re-export the above plus `make_backend(name: str, **kw)` for
`"sim"` and `"anthropic"`.

### 4.4 `evorg.bench`

`evorg/bench/tasks.py`:

```python
@dataclass(frozen=True)
class Task:
    id: str
    title: str
    prompt: str                     # natural-language spec
    signature: str                  # e.g. "def rle_encode(s: str) -> str:"
    visible_tests: tuple[str, ...]  # assert statements shown to the organization
    hidden_tests: tuple[str, ...]   # assert statements used only for scoring
    reference_solution: str
    buggy_solutions: tuple[str, ...]
    difficulty: float               # 0.0 (trivial) .. 3.0 (hard)
    split: str                      # "train" | "holdout"

def load_benchmark(split: str = "train") -> list[Task]   # "train" | "holdout" | "all"
def get_task(task_id: str) -> Task
def task_index() -> dict[str, Task]
```

Twelve tasks: eight `train`, four `holdout`, a spread of difficulty. Each task has at least
two buggy solutions: at least one that fails a visible test, and at least one **subtle**
bug that passes every visible test but fails a hidden one. The subtle bugs are what make a
reviewer worth paying for.

`evorg/bench/grader.py`:

```python
@dataclass(frozen=True)
class GradeResult:
    passed: bool                    # all tests passed
    n_passed: int
    n_total: int
    failures: tuple[str, ...]       # the failing assert lines (truncated to 200 chars)
    error: str | None               # syntax error, timeout, crash

def extract_code(text: str) -> str              # first ```python block, else whole text
def grade(code: str, tests: Sequence[str], timeout_s: float = 5.0) -> GradeResult
```

`grade` runs the code and tests in a fresh `python -I` subprocess with a timeout. Each test
runs in its own try/except so one failure does not hide others. Results come back as JSON
on stdout. This is isolation from the evolver process, not a security sandbox; the README
says so.

### 4.5 `evorg.executor`

```python
@dataclass(frozen=True)
class Turn:
    role: str; kind: str; text: str

@dataclass(frozen=True)
class TaskResult:
    task_id: str
    passed: bool
    pass_fraction: float            # over visible + hidden tests
    attempts: int
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    transcript: tuple[Turn, ...]

def run_task(genome: OrgGenome, task: Task, backend: LLMBackend) -> TaskResult
```

Flow: every planner runs once (in role order) and the plans are joined. Then up to
`1 + max_retries` attempts: the coder writes code from the task prompt, signature, visible
tests, `PLAN:` and, after the first attempt, `FEEDBACK:`. If the org has a tester, the
executor grades the code on the visible tests; on failure the tester turns the failures into
feedback. Else if the org has a reviewer, the reviewer is asked; `REVISE` becomes feedback.
No feedback means the attempt is final. An org with no tester and no reviewer therefore
cannot use its retries, which is intended: retries need a source of signal. The final code
is graded on visible + hidden tests.

### 4.6 `evorg.mutation`

```python
class MutationError(ValueError): ...

@dataclass(frozen=True)
class Mutation:
    op: str      # add_instruction | remove_instruction | replace_instruction | add_role
                 # | remove_role | change_model | set_retries
    role: str | None = None
    value: str | int | None = None   # instruction text, new kind, model tier, retry count
    index: int | None = None         # instruction index for remove/replace
    rationale: str = ""
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> "Mutation"
    def describe(self) -> str        # one line for lineage reports

def apply_mutation(genome: OrgGenome, m: Mutation) -> OrgGenome
    # returns a validated child with parent_ids=(genome.genome_id,),
    # generation=genome.generation + 1, mutation_note=m.describe(); raises MutationError
def random_mutation(genome: OrgGenome, rng: random.Random) -> Mutation   # always valid

class Architect:
    def __init__(self, backend: LLMBackend, rng: random.Random, model: str = "large")
    def propose(self, genome: OrgGenome, report: str, n: int = 1) -> list[Mutation]
```

`add_role` uses `value` as the new role's kind and `role` as its name. `Architect.propose`
sends the genome JSON, the evaluation report and the mutation schema to the backend
(`role_kind="architect"`, `metadata={"genome": genome.to_dict(), "n": n}`), parses JSON
leniently (a fenced block or the first `{...}`), drops anything that does not apply, and tops
up with `random_mutation` so it always returns exactly `n`. It records how many proposals
were valid in `self.stats`.

### 4.7 `evorg.evolution`

```python
@dataclass(frozen=True)
class EvolutionConfig:
    population: int = 6
    generations: int = 5
    tournament_size: int = 3
    elite: int = 1
    cost_weight: float = 10.0       # fitness = pass_rate − cost_weight · mean cost per task
    seed: int = 0

@dataclass(frozen=True)
class Evaluation:
    genome_id: str
    pass_rate: float
    mean_cost_usd: float
    fitness: float
    results: tuple[TaskResult, ...]
    def report(self) -> str          # short failure summary handed to the architect

@dataclass
class GenerationRecord:
    index: int
    members: list[tuple[str, float]]   # (genome_id, fitness), best first
    best_id: str

@dataclass
class RunResult:
    config: EvolutionConfig
    backend: str
    generations: list[GenerationRecord]
    genomes: dict[str, OrgGenome]      # every genome ever evaluated
    evaluations: dict[str, Evaluation]  # train evaluations
    champion_id: str
    seed_id: str
    holdout: dict[str, Evaluation]      # champion and seed on held-out tasks
    architect_stats: dict
    def lineage(self, genome_id: str) -> list[OrgGenome]   # seed ... genome
    def to_dict(self) -> dict

def evaluate(genome, tasks, backend, cost_weight) -> Evaluation
def evolve(config, backend, train, holdout, architect, on_generation=None) -> RunResult
```

Generation 0 is the seed plus `population − 1` single mutations of it. Each generation, every
parent slot is filled by tournament selection; each parent produces one child through the
architect; children are evaluated (evaluations are cached by `genome_id`, since identical
organizations behave identically); the next population is the `elite` best plus tournament
winners from the rest of parents and children, without duplicates. Ties break on lower cost,
then `genome_id`, so runs are reproducible.

### 4.8 `evorg.report` and `evorg.cli`

`report.render_markdown(run: RunResult) -> str`: fitness per generation (table and a unicode
sparkline), the champion's genome, `diff_genomes(seed, champion)`, the lineage chain with
each mutation's rationale, and the held-out comparison of seed vs champion.

`evorg run --backend sim|anthropic --generations N --population N --seed N --out DIR`
writes `DIR/run.json` and `DIR/report.md` and prints the report. `evorg show DIR` prints a
saved report. `evorg bench GENOME.json [--split holdout]` evaluates one genome.

## 5. Testing strategy

* Unit tests per module, no network, no API key.
* Benchmark invariants: every reference solution passes all of its tests; every buggy
  solution fails at least one; every task has a subtle bug that passes all visible tests.
* Determinism: two runs with the same seed produce identical `run.json`.
* An end-to-end test runs a 2-generation evolution on the simulated backend.
* CI runs lint, tests on Python 3.11 to 3.13, and a smoke run of the CLI.

## 6. Honest limits

The simulated backend is a model of capability, written by hand. Results on it show that the
machinery works and that selection finds the structures the model rewards; they say nothing
about real LLMs. The interesting experiment is the Anthropic backend, which uses the same
loop. The grader isolates code in a subprocess but is not a security boundary.

## 7. Milestone 2

FastAPI service and React lineage explorer; real-model experiment write-up; crossover
between organizations; Pareto selection on pass rate and cost; parallel evaluation.
