"""Tests for evorg.mutation: applying edits, invalid edits, serialisation and random draws."""

from __future__ import annotations

import json
import random
from dataclasses import replace

import pytest

from evorg.genes import INSTRUCTION_POOL
from evorg.genome import OrgGenome, RoleSpec, seed_genome
from evorg.llm.base import LLMRequest, LLMResponse
from evorg.llm.simulated import SimulatedBackend
from evorg.mutation import (
    OPS,
    Architect,
    Mutation,
    MutationError,
    apply_mutation,
    random_mutation,
)


def role(name: str, kind: str, *instructions: str, model: str = "small") -> RoleSpec:
    return RoleSpec(name=name, kind=kind, instructions=tuple(instructions), model=model)


def rich_genome() -> OrgGenome:
    """Five roles (two planners), instructions on several roles, retries at 2."""
    return OrgGenome(
        roles=(
            role("planner", "planner", "Work through the problem step by step before coding."),
            role(
                "coder",
                "coder",
                "Write a correct Python function.",
                "Use type hints.",
                model="large",
            ),
            role("tester", "tester"),
            role("reviewer", "reviewer", "Consider edge cases such as empty input."),
            role("planner2", "planner"),
        ),
        max_retries=2,
        parent_ids=("aaaaaaaaaa",),
        generation=3,
    )


def planner_coder() -> OrgGenome:
    return OrgGenome(roles=(role("planner", "planner"), role("coder", "coder")))


def mutated(genome: OrgGenome, m: Mutation) -> OrgGenome:
    child = apply_mutation(genome, m)
    child.validate()
    return child


# --- ops -------------------------------------------------------------------------------


def test_ops_are_the_seven_named_ops():
    assert OPS == (
        "add_instruction",
        "remove_instruction",
        "replace_instruction",
        "add_role",
        "remove_role",
        "change_model",
        "set_retries",
    )


def test_add_instruction_appends_and_sets_lineage():
    seed = seed_genome()
    m = Mutation(
        "add_instruction",
        role="coder",
        value="Handle empty input explicitly.",
        rationale="why",
    )
    child = mutated(seed, m)
    assert child.role("coder").instructions == (
        "Write a correct Python function.",
        "Handle empty input explicitly.",
    )
    assert child.parent_ids == (seed.genome_id,)
    assert child.generation == seed.generation + 1
    assert child.mutation_note == "coder: + instruction 'Handle empty input explicitly.'"
    assert child.genome_id != seed.genome_id


def test_lineage_increments_from_a_later_generation():
    parent = rich_genome()
    child = mutated(parent, Mutation("set_retries", value=3))
    assert child.generation == parent.generation + 1
    assert child.parent_ids == (parent.genome_id,)


def test_remove_instruction_drops_the_indexed_one():
    child = mutated(rich_genome(), Mutation("remove_instruction", role="coder", index=0))
    assert child.role("coder").instructions == ("Use type hints.",)


def test_replace_instruction_swaps_text_in_place():
    child = mutated(
        rich_genome(),
        Mutation("replace_instruction", role="coder", index=1, value="Write a short docstring."),
    )
    assert child.role("coder").instructions == (
        "Write a correct Python function.",
        "Write a short docstring.",
    )


def test_add_role_with_no_name_uses_the_kind():
    child = mutated(seed_genome(), Mutation("add_role", value="tester"))
    assert [r.name for r in child.roles] == ["coder", "tester"]
    added = child.role("tester")
    assert (added.kind, added.model, added.instructions) == ("tester", "small", ())
    assert child.mutation_note == "+ role tester (tester)"


def test_add_role_with_no_name_uses_kind_two_when_kind_name_is_taken():
    child = mutated(planner_coder(), Mutation("add_role", value="planner"))
    assert [r.name for r in child.roles] == ["planner", "planner2", "coder"]
    assert child.mutation_note == "+ role planner2 (planner)"


def test_add_role_with_no_name_fails_when_both_names_are_taken():
    with pytest.raises(MutationError):
        apply_mutation(rich_genome(), Mutation("add_role", value="planner"))


def test_add_role_planner_is_inserted_before_the_coder():
    child = mutated(seed_genome(), Mutation("add_role", value="planner"))
    assert [r.name for r in child.roles] == ["planner", "coder"]
    second = mutated(child, Mutation("add_role", value="planner", role="scout"))
    assert [r.name for r in second.roles] == ["planner", "scout", "coder"]


def test_add_role_tester_and_reviewer_are_appended_at_the_end():
    child = mutated(planner_coder(), Mutation("add_role", value="reviewer", role="critic"))
    assert [r.name for r in child.roles] == ["planner", "coder", "critic"]
    assert child.role("critic").kind == "reviewer"


def test_remove_role_removes_a_named_non_coder():
    child = mutated(rich_genome(), Mutation("remove_role", role="tester"))
    assert [r.name for r in child.roles] == ["planner", "coder", "reviewer", "planner2"]


def test_change_model_sets_the_named_tier():
    child = mutated(seed_genome(), Mutation("change_model", role="coder", value="large"))
    assert child.role("coder").model == "large"
    assert child.mutation_note == "coder: model -> large"


def test_change_model_without_value_flips_the_tier():
    up = mutated(seed_genome(), Mutation("change_model", role="coder"))
    assert up.role("coder").model == "large"
    down = mutated(rich_genome(), Mutation("change_model", role="coder"))
    assert down.role("coder").model == "small"
    assert down.mutation_note == "coder: model -> small"


def test_set_retries_changes_the_budget():
    child = mutated(seed_genome(), Mutation("set_retries", value=2))
    assert child.max_retries == 2
    assert child.mutation_note == "max_retries -> 2"


def test_set_retries_accepts_numeric_string_after_from_dict():
    m = Mutation.from_dict({"op": "set_retries", "value": "0"})
    assert mutated(rich_genome(), m).max_retries == 0


def test_rationale_is_kept_but_not_part_of_describe():
    m = Mutation("set_retries", value=2, rationale="cheap extra check")
    assert m.describe() == "max_retries -> 2"
    assert mutated(seed_genome(), m).mutation_note == "max_retries -> 2"


def test_parent_is_not_modified():
    parent = rich_genome()
    apply_mutation(parent, Mutation("remove_role", role="tester"))
    assert [r.name for r in parent.roles] == [
        "planner",
        "coder",
        "tester",
        "reviewer",
        "planner2",
    ]


# --- invalid edits ----------------------------------------------------------------------


def _invalid_cases() -> list:
    seed = seed_genome()
    rich = rich_genome()
    eight = replace(seed, roles=(role("coder", "coder", *INSTRUCTION_POOL[:8]),))
    return [
        pytest.param(seed, Mutation("add_instruction", role="ghost", value="Be careful."),
                     id="add-missing-role"),
        pytest.param(seed, Mutation("add_instruction", role=None, value="Be careful."),
                     id="add-role-none"),
        pytest.param(seed, Mutation("add_instruction", role="coder", value="   "),
                     id="add-blank-text"),
        pytest.param(seed, Mutation("add_instruction", role="coder", value=5),
                     id="add-non-str-text"),
        pytest.param(seed, Mutation("add_instruction", role="coder", value="x" * 201),
                     id="add-too-long"),
        pytest.param(rich, Mutation("add_instruction", role="coder", value="Use type hints."),
                     id="add-duplicate"),
        pytest.param(eight, Mutation("add_instruction", role="coder", value="Be brief."),
                     id="add-ninth-instruction"),
        pytest.param(rich, Mutation("remove_instruction", role="coder", index=5),
                     id="remove-index-high"),
        pytest.param(rich, Mutation("remove_instruction", role="coder", index=-1),
                     id="remove-index-negative"),
        pytest.param(rich, Mutation("remove_instruction", role="coder"),
                     id="remove-index-missing"),
        pytest.param(rich, Mutation("remove_instruction", role="ghost", index=0),
                     id="remove-missing-role"),
        pytest.param(rich, Mutation("replace_instruction", role="coder", index=9,
                                    value="Be careful."), id="replace-index-high"),
        pytest.param(rich, Mutation("replace_instruction", role="coder", index=0, value=3),
                     id="replace-non-str"),
        pytest.param(rich, Mutation("replace_instruction", role="coder", index=0,
                                    value="Write a correct Python function."),
                     id="replace-same-text"),
        pytest.param(rich, Mutation("add_role", value="coder"), id="second-coder"),
        pytest.param(rich, Mutation("add_role", value="wizard", role="w"), id="unknown-kind"),
        pytest.param(rich, Mutation("add_role", value=None, role="w"), id="kind-missing"),
        pytest.param(rich, Mutation("add_role", role="coder", value="tester"),
                     id="duplicate-name"),
        pytest.param(rich, Mutation("add_role", role="  ", value="tester"), id="blank-name"),
        pytest.param(rich, Mutation("add_role", role="p3", value="planner"),
                     id="third-planner"),
        pytest.param(seed, Mutation("remove_role", role="coder"), id="remove-coder"),
        pytest.param(rich, Mutation("remove_role", role="ghost"), id="remove-missing-role-2"),
        pytest.param(rich, Mutation("change_model", role="coder", value="large"),
                     id="model-noop"),
        pytest.param(seed, Mutation("change_model", role="coder", value="small"),
                     id="model-noop-small"),
        pytest.param(seed, Mutation("change_model", role="coder", value="huge"),
                     id="model-unknown"),
        pytest.param(seed, Mutation("change_model", role="ghost", value="large"),
                     id="model-missing-role"),
        pytest.param(seed, Mutation("set_retries", value=5), id="retries-high"),
        pytest.param(seed, Mutation("set_retries", value=-1), id="retries-negative"),
        pytest.param(rich, Mutation("set_retries", value=2), id="retries-noop"),
        pytest.param(seed, Mutation("set_retries", value="two"), id="retries-not-int"),
        pytest.param(seed, Mutation("set_retries", value=True), id="retries-bool"),
        pytest.param(seed, Mutation("teleport", role="coder"), id="unknown-op"),
        pytest.param(seed, Mutation(""), id="empty-op"),
    ]


@pytest.mark.parametrize(("genome", "m"), _invalid_cases())
def test_invalid_mutations_raise(genome, m):
    with pytest.raises(MutationError):
        apply_mutation(genome, m)


def test_mutation_error_is_a_value_error():
    assert issubclass(MutationError, ValueError)


def test_third_planner_error_names_the_limit():
    with pytest.raises(MutationError, match="at most 2 planner"):
        apply_mutation(rich_genome(), Mutation("add_role", role="p3", value="planner"))


def test_no_op_mutation_is_refused_with_a_clear_message():
    with pytest.raises(MutationError, match="changes nothing"):
        apply_mutation(seed_genome(), Mutation("set_retries", value=1))


# --- serialisation ----------------------------------------------------------------------


def test_to_dict_from_dict_round_trip():
    cases = [
        Mutation("add_instruction", role="coder", value="Use type hints.", rationale="r"),
        Mutation("remove_instruction", role="coder", index=0),
        Mutation("replace_instruction", role="coder", index=1, value="x", rationale="y"),
        Mutation("add_role", role="tester", value="tester"),
        Mutation("remove_role", role="tester"),
        Mutation("change_model", role="coder", value="large"),
        Mutation("set_retries", value=3),
    ]
    for m in cases:
        assert Mutation.from_dict(m.to_dict()) == m


def test_from_dict_tolerates_missing_keys_and_loose_ints():
    m = Mutation.from_dict({"op": "remove_instruction", "role": "coder", "index": "1"})
    assert (m.op, m.role, m.index, m.value, m.rationale) == (
        "remove_instruction",
        "coder",
        1,
        None,
        "",
    )
    assert Mutation.from_dict({"op": "set_retries", "value": 2.0}).value == 2
    assert Mutation.from_dict({"op": "remove_instruction", "index": "soon"}).index is None


def test_from_dict_without_op_gives_an_op_that_fails_to_apply():
    m = Mutation.from_dict({"role": "coder"})
    with pytest.raises(MutationError):
        apply_mutation(seed_genome(), m)


def test_from_dict_rejects_non_objects():
    with pytest.raises(MutationError):
        Mutation.from_dict("add a tester")  # type: ignore[arg-type]


def test_describe_lines():
    add = Mutation("add_instruction", role="coder", value="Handle empty input explicitly.")
    assert add.describe() == "coder: + instruction 'Handle empty input explicitly.'"
    assert Mutation("add_role", role="tester", value="tester").describe() == (
        "+ role tester (tester)"
    )
    assert Mutation("remove_role", role="planner").describe() == "- role planner"
    assert Mutation("change_model", role="coder", value="large").describe() == (
        "coder: model -> large"
    )
    assert Mutation("set_retries", value=2).describe() == "max_retries -> 2"
    replace_m = Mutation("replace_instruction", role="coder", index=0, value="Be clear.")
    assert replace_m.describe() == "coder: replace instruction 0 with 'Be clear.'"


# --- random mutation --------------------------------------------------------------------


def _starts() -> list[OrgGenome]:
    full_coder = replace(
        seed_genome(), roles=(role("coder", "coder", *INSTRUCTION_POOL[:8]),), max_retries=4
    )
    return [seed_genome(), rich_genome(), planner_coder(), full_coder]


@pytest.mark.parametrize("start", range(4))
def test_random_mutation_always_applies(start):
    genome = _starts()[start]
    for seed in range(200):
        m = random_mutation(genome, random.Random(seed))
        child = apply_mutation(genome, m)
        child.validate()
        assert m.rationale == "random exploration"


def test_random_mutation_covers_several_ops():
    rng = random.Random(7)
    ops = {random_mutation(rich_genome(), rng).op for _ in range(300)}
    assert len(ops) >= 6


def test_random_mutation_is_deterministic_for_a_seed():
    a = random_mutation(rich_genome(), random.Random(3))
    b = random_mutation(rich_genome(), random.Random(3))
    assert a == b


def test_random_mutation_returns_resolved_fields():
    for seed in range(100):
        m = random_mutation(planner_coder(), random.Random(seed))
        if m.op == "change_model":
            assert m.value in ("small", "large")
        if m.op == "add_role":
            assert m.role is not None


# --- architect ----------------------------------------------------------------------------


class FakeBackend:
    """Returns the scripted replies in order, one per call, and records the requests."""

    name = "fake"

    def __init__(self, *replies: str, cost: float = 0.01) -> None:
        self.replies = list(replies)
        self.cost = cost
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        text = self.replies.pop(0) if self.replies else ""
        return LLMResponse(text=text, input_tokens=10, output_tokens=5, cost_usd=self.cost)


ADD_EMPTY = {
    "op": "add_instruction",
    "role": "coder",
    "value": "Handle empty input explicitly.",
    "index": None,
    "rationale": "empty input is a common edge case",
}
SET_RETRIES_2 = {"op": "set_retries", "role": None, "value": 2, "index": None, "rationale": "r"}
GHOST = {"op": "remove_role", "role": "ghost", "value": None, "index": None, "rationale": ""}


def _assert_applies(genome: OrgGenome, mutations: list[Mutation]) -> None:
    for m in mutations:
        apply_mutation(genome, m)


def test_architect_request_carries_genome_role_kind_and_n():
    backend = FakeBackend(json.dumps({"mutations": [ADD_EMPTY]}))
    genome = rich_genome()
    architect = Architect(backend, random.Random(0))
    architect.propose(genome, "pass rate 0.5", n=1)
    request = backend.requests[0]
    assert request.role_kind == "architect"
    assert request.model == "large"
    assert request.metadata["genome"] == genome.to_dict()
    assert request.metadata["n"] == 1
    assert "Propose exactly 1 mutations" in request.prompt
    assert "pass rate 0.5" in request.prompt
    assert json.dumps(genome.to_dict(), indent=2) in request.prompt
    for op in OPS:
        assert op in request.prompt
    assert "architect" in request.system


def test_architect_model_is_configurable():
    backend = FakeBackend()
    Architect(backend, random.Random(0), model="small").propose(seed_genome(), "", n=1)
    assert backend.requests[0].model == "small"


@pytest.mark.parametrize("seed", range(30))
def test_simulated_architect_with_one_mutation_always_applies(seed):
    genome = seed_genome()
    for parent in (genome, rich_genome()):
        architect = Architect(SimulatedBackend(task_index={}, seed=seed), random.Random(seed))
        out = architect.propose(parent, "report", n=1)
        assert len(out) == 1
        _assert_applies(parent, out)
        assert architect.stats["calls"] == 1
        assert architect.stats["valid"] == architect.stats["proposed"] == 1
        assert architect.stats["fallback"] == 0
        assert architect.stats["cost_usd"] > 0


def test_simulated_architect_with_several_mutations_still_returns_exactly_n():
    architect = Architect(SimulatedBackend(task_index={}, seed=4), random.Random(4))
    out = architect.propose(rich_genome(), "report", n=3)
    assert len(out) == 3
    _assert_applies(rich_genome(), out)


def test_garbage_reply_falls_back_to_random_mutations():
    backend = FakeBackend("I think you should add a tester!")
    architect = Architect(backend, random.Random(1))
    out = architect.propose(seed_genome(), "report", n=2)
    assert len(out) == 2
    _assert_applies(seed_genome(), out)
    assert architect.stats["proposed"] == 0
    assert architect.stats["valid"] == 0
    assert architect.stats["fallback"] == 2


def test_fenced_json_block_is_parsed():
    reply = (
        "Here is my proposal:\n```json\n"
        + json.dumps({"mutations": [ADD_EMPTY]})
        + "\n```\nGood luck."
    )
    architect = Architect(FakeBackend(reply), random.Random(0))
    out = architect.propose(seed_genome(), "report", n=1)
    assert out[0].op == "add_instruction"
    assert out[0].value == "Handle empty input explicitly."
    assert architect.stats["fallback"] == 0


def test_bare_fence_without_language_is_parsed():
    reply = "```\n" + json.dumps([SET_RETRIES_2]) + "\n```"
    architect = Architect(FakeBackend(reply), random.Random(0))
    out = architect.propose(seed_genome(), "report", n=1)
    assert out[0].op == "set_retries" and out[0].value == 2
    assert architect.stats["fallback"] == 0


def test_bare_list_is_accepted():
    architect = Architect(FakeBackend(json.dumps([SET_RETRIES_2])), random.Random(0))
    out = architect.propose(seed_genome(), "report", n=1)
    assert out == [Mutation.from_dict(SET_RETRIES_2)]
    assert architect.stats["fallback"] == 0


def test_json_inside_prose_is_found_by_braces():
    reply = "Sure! " + json.dumps({"mutations": [SET_RETRIES_2]}) + " Thanks."
    architect = Architect(FakeBackend(reply), random.Random(0))
    assert architect.propose(seed_genome(), "report", n=1)[0].value == 2


def test_partially_invalid_list_keeps_the_valid_ones():
    reply = json.dumps({"mutations": [GHOST, "junk", SET_RETRIES_2, ADD_EMPTY]})
    architect = Architect(FakeBackend(reply), random.Random(0))
    out = architect.propose(seed_genome(), "report", n=2)
    assert [m.op for m in out] == ["set_retries", "add_instruction"]
    _assert_applies(seed_genome(), out)
    assert architect.stats["proposed"] == 4
    assert architect.stats["valid"] == 2
    assert architect.stats["fallback"] == 0


def test_shortfall_is_topped_up_to_exactly_n():
    reply = json.dumps({"mutations": [GHOST, SET_RETRIES_2]})
    architect = Architect(FakeBackend(reply), random.Random(2))
    out = architect.propose(seed_genome(), "report", n=3)
    assert len(out) == 3
    _assert_applies(seed_genome(), out)
    assert out[0].op == "set_retries"
    assert architect.stats["valid"] == 1
    assert architect.stats["fallback"] == 2


def test_duplicate_proposals_yield_distinct_children():
    reply = json.dumps({"mutations": [SET_RETRIES_2, SET_RETRIES_2]})
    architect = Architect(FakeBackend(reply), random.Random(5))
    out = architect.propose(seed_genome(), "report", n=2)
    ids = {apply_mutation(seed_genome(), m).genome_id for m in out}
    assert len(ids) == 2
    assert architect.stats["valid"] == 2
    assert architect.stats["fallback"] == 1


def test_stats_accumulate_across_calls():
    backend = FakeBackend(
        json.dumps({"mutations": [SET_RETRIES_2]}),
        "no json here",
        cost=0.25,
    )
    architect = Architect(backend, random.Random(0))
    architect.propose(seed_genome(), "report", n=1)
    architect.propose(seed_genome(), "report", n=1)
    assert architect.stats["calls"] == 2
    assert architect.stats["proposed"] == 1
    assert architect.stats["valid"] == 1
    assert architect.stats["fallback"] == 1
    assert architect.stats["cost_usd"] == pytest.approx(0.5)


def test_non_list_mutations_value_is_ignored():
    reply = '{"mutations": 5}'
    architect = Architect(FakeBackend(reply), random.Random(0))
    out = architect.propose(seed_genome(), "report", n=1)
    assert len(out) == 1
    assert architect.stats["proposed"] == 0
    assert architect.stats["fallback"] == 1


def test_proposal_that_only_applies_as_a_chain_is_dropped():
    # The second item only makes sense after the first; applied alone it is refused.
    chain = [
        {"op": "add_role", "role": "tester", "value": "tester"},
        {"op": "add_instruction", "role": "tester", "value": "Check edge cases."},
    ]
    architect = Architect(FakeBackend(json.dumps({"mutations": chain})), random.Random(0))
    out = architect.propose(seed_genome(), "report", n=2)
    assert out[0].op == "add_role"
    assert architect.stats["valid"] == 1
    assert architect.stats["fallback"] == 1


def test_add_role_without_name_is_named_before_it_is_returned():
    proposal = {"op": "add_role", "role": None, "value": "planner", "index": None}
    architect = Architect(FakeBackend(json.dumps({"mutations": [proposal]})), random.Random(0))
    out = architect.propose(seed_genome(), "report", n=1)
    assert out[0].role == "planner"
    assert out[0].describe() == "+ role planner (planner)"


def test_propose_rejects_non_positive_n():
    with pytest.raises(ValueError):
        Architect(FakeBackend(), random.Random(0)).propose(seed_genome(), "", n=0)
