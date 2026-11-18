"""Tests for evorg.genome: validation, identity, serialisation, prompts and diffs."""

from __future__ import annotations

import pytest

from evorg.genome import (
    GenomeError,
    OrgGenome,
    RoleSpec,
    diff_genomes,
    seed_genome,
)


def coder(**kw) -> RoleSpec:
    base = {"name": "coder", "kind": "coder", "instructions": (), "model": "small"}
    base.update(kw)
    return RoleSpec(**base)


def org(*roles: RoleSpec, **kw) -> OrgGenome:
    return OrgGenome(roles=roles or (coder(),), **kw)


def bad(genome: OrgGenome, fragment: str) -> None:
    with pytest.raises(GenomeError, match=fragment):
        genome.validate()


# --- seed and validation -------------------------------------------------------------


def test_seed_genome_validates_and_has_expected_shape():
    seed = seed_genome()
    seed.validate()
    assert [r.name for r in seed.roles] == ["coder"]
    assert seed.roles[0].kind == "coder"
    assert seed.roles[0].model == "small"
    assert seed.roles[0].instructions == ("Write a correct Python function.",)
    assert seed.max_retries == 1
    assert seed.generation == 0
    assert seed.parent_ids == ()


def test_full_valid_organization_passes():
    genome = org(
        RoleSpec("planner", "planner", ("Plan first.",), "small"),
        RoleSpec("planner2", "planner", (), "large"),
        coder(instructions=("a", "b")),
        RoleSpec("tester", "tester", (), "small"),
        RoleSpec("reviewer", "reviewer", (), "large"),
        max_retries=4,
    )
    genome.validate()


def test_rejects_zero_roles():
    with pytest.raises(GenomeError, match="1 to 6 roles, got 0"):
        OrgGenome(roles=()).validate()


def test_rejects_more_than_six_roles():
    roles = [coder(name="coder")]
    roles += [RoleSpec(f"p{i}", "planner") for i in range(2)]
    roles += [RoleSpec("t", "tester"), RoleSpec("r", "reviewer")]
    roles += [RoleSpec("x1", "planner"), RoleSpec("x2", "planner")]
    assert len(roles) == 7
    bad(OrgGenome(roles=tuple(roles)), "1 to 6 roles, got 7")


def test_rejects_blank_role_name():
    bad(org(coder(name="   ")), "non-empty")


def test_rejects_empty_role_name():
    bad(org(coder(name="")), "non-empty")


def test_rejects_duplicate_role_names():
    genome = org(
        coder(),
        RoleSpec("coder", "planner"),
    )
    bad(genome, "duplicate role name 'coder'")


def test_rejects_unknown_kind():
    genome = org(coder(), RoleSpec("boss", "manager"))
    bad(genome, "unknown kind 'manager'")


def test_rejects_unknown_model():
    bad(org(coder(model="medium")), "unknown model 'medium'")


def test_rejects_missing_coder():
    genome = org(RoleSpec("planner", "planner"))
    bad(genome, "exactly one coder, got 0")


def test_rejects_two_coders():
    genome = org(coder(), RoleSpec("coder2", "coder"))
    bad(genome, "exactly one coder, got 2")


def test_rejects_three_planners():
    genome = org(
        coder(),
        RoleSpec("p1", "planner"),
        RoleSpec("p2", "planner"),
        RoleSpec("p3", "planner"),
    )
    bad(genome, "at most 2 planner allowed, got 3")


def test_rejects_two_testers():
    genome = org(coder(), RoleSpec("t1", "tester"), RoleSpec("t2", "tester"))
    bad(genome, "at most 1 tester allowed, got 2")


def test_rejects_two_reviewers():
    genome = org(coder(), RoleSpec("r1", "reviewer"), RoleSpec("r2", "reviewer"))
    bad(genome, "at most 1 reviewer allowed, got 2")


def test_rejects_nine_instructions():
    instructions = tuple(f"instruction {i}" for i in range(9))
    bad(org(coder(instructions=instructions)), "9 instructions; at most 8")


def test_accepts_eight_instructions():
    org(coder(instructions=tuple(f"instruction {i}" for i in range(8)))).validate()


def test_rejects_empty_instruction():
    bad(org(coder(instructions=("",))), "1 to 200 characters")


def test_rejects_blank_instruction():
    bad(org(coder(instructions=("   ",))), "not blank")


def test_rejects_instruction_over_200_characters():
    bad(org(coder(instructions=("x" * 201,))), "201 characters")


def test_accepts_instruction_of_exactly_200_characters():
    org(coder(instructions=("x" * 200,))).validate()


def test_rejects_duplicate_instruction_within_role():
    bad(org(coder(instructions=("same", "same"))), "repeats the instruction 'same'")


def test_same_instruction_in_different_roles_is_allowed():
    genome = org(
        coder(instructions=("shared",)),
        RoleSpec("reviewer", "reviewer", ("shared",)),
    )
    genome.validate()


@pytest.mark.parametrize("retries", [-1, 5, 100])
def test_rejects_retries_out_of_range(retries):
    bad(org(coder(), max_retries=retries), "max_retries must be between 0 and 4")


@pytest.mark.parametrize("retries", [0, 1, 4])
def test_accepts_retries_in_range(retries):
    org(coder(), max_retries=retries).validate()


# --- lookup helpers ------------------------------------------------------------------


def test_role_lookup_by_name():
    genome = org(coder(), RoleSpec("reviewer", "reviewer"))
    assert genome.role("reviewer").kind == "reviewer"


def test_role_lookup_missing_raises_key_error():
    with pytest.raises(KeyError):
        org(coder()).role("missing")


def test_roles_of_kind_returns_matching_roles_in_order():
    genome = org(
        RoleSpec("p1", "planner"),
        coder(),
        RoleSpec("p2", "planner"),
    )
    assert [r.name for r in genome.roles_of_kind("planner")] == ["p1", "p2"]
    assert genome.roles_of_kind("tester") == ()


# --- genome_id -----------------------------------------------------------------------


def test_genome_id_is_ten_hex_chars_and_stable():
    seed = seed_genome()
    gid = seed.genome_id
    assert len(gid) == 10
    assert all(c in "0123456789abcdef" for c in gid)
    assert seed_genome().genome_id == gid


def test_genome_id_ignores_lineage_fields():
    base = seed_genome()
    child = OrgGenome(
        roles=base.roles,
        max_retries=base.max_retries,
        parent_ids=("abcdef0123", "fedcba9876"),
        generation=7,
        mutation_note="added a reviewer on a whim",
    )
    assert child.genome_id == base.genome_id


def test_genome_id_changes_when_instruction_changes():
    base = seed_genome()
    edited = OrgGenome(
        roles=(
            RoleSpec(
                name="coder",
                kind="coder",
                instructions=("Write a correct Python function, carefully.",),
                model="small",
            ),
        ),
        max_retries=1,
    )
    assert edited.genome_id != base.genome_id


def test_genome_id_changes_with_model_retries_and_roles():
    base = seed_genome()
    large = OrgGenome(roles=(coder(model="large", instructions=base.roles[0].instructions),))
    more_retries = OrgGenome(roles=base.roles, max_retries=2)
    with_reviewer = OrgGenome(roles=(*base.roles, RoleSpec("reviewer", "reviewer")))
    ids = {base.genome_id, large.genome_id, more_retries.genome_id, with_reviewer.genome_id}
    assert len(ids) == 4


def test_genome_id_depends_on_instruction_order():
    a = org(coder(instructions=("one", "two")))
    b = org(coder(instructions=("two", "one")))
    assert a.genome_id != b.genome_id


# --- serialisation -------------------------------------------------------------------


def test_round_trip_is_exact():
    genome = org(
        RoleSpec("planner", "planner", ("Plan.", "Be careful."), "large"),
        coder(instructions=("Write code.",)),
        RoleSpec("reviewer", "reviewer", (), "small"),
        max_retries=3,
        parent_ids=("aaaaaaaaaa", "bbbbbbbbbb"),
        generation=4,
        mutation_note="add_role reviewer",
    )
    restored = OrgGenome.from_dict(genome.to_dict())
    assert restored == genome
    assert restored.roles[0].instructions == ("Plan.", "Be careful.")
    assert restored.parent_ids == ("aaaaaaaaaa", "bbbbbbbbbb")


def test_to_dict_contains_all_fields_with_lists_and_genome_id():
    genome = seed_genome()
    d = genome.to_dict()
    assert d["genome_id"] == genome.genome_id
    assert d["roles"] == [
        {
            "name": "coder",
            "kind": "coder",
            "instructions": ["Write a correct Python function."],
            "model": "small",
        }
    ]
    assert d["max_retries"] == 1
    assert d["parent_ids"] == []
    assert d["generation"] == 0
    assert d["mutation_note"] == ""


def test_from_dict_ignores_stored_genome_id():
    d = seed_genome().to_dict()
    d["genome_id"] = "0000000000"
    assert OrgGenome.from_dict(d).genome_id == seed_genome().genome_id


def test_from_dict_converts_lists_to_tuples():
    d = seed_genome().to_dict()
    d["parent_ids"] = ["x"]
    restored = OrgGenome.from_dict(d)
    assert isinstance(restored.roles, tuple)
    assert isinstance(restored.roles[0].instructions, tuple)
    assert isinstance(restored.parent_ids, tuple)


def test_genomes_are_hashable_and_frozen():
    genome = seed_genome()
    assert hash(genome) == hash(OrgGenome.from_dict(genome.to_dict()))
    with pytest.raises(AttributeError):
        genome.max_retries = 3  # type: ignore[misc]
    with pytest.raises(AttributeError):
        genome.roles[0].model = "large"  # type: ignore[misc]


# --- system prompts ------------------------------------------------------------------


def test_system_prompt_without_instructions_is_just_the_persona():
    prompt = RoleSpec("reviewer", "reviewer").system_prompt()
    assert "reviewer" in prompt.lower()
    assert "Instructions:" not in prompt
    assert "- " not in prompt
    assert "\n" not in prompt


def test_system_prompt_contains_instructions_as_bullets():
    role = RoleSpec(
        "coder",
        "coder",
        ("Use type hints.", "Handle empty input explicitly."),
    )
    prompt = role.system_prompt()
    lines = prompt.split("\n")
    assert lines[0].startswith("You are the coder")
    assert lines[1] == "Instructions:"
    assert lines[2] == "- Use type hints."
    assert lines[3] == "- Handle empty input explicitly."
    assert len(lines) == 4


@pytest.mark.parametrize("kind", ["planner", "coder", "tester", "reviewer"])
def test_each_kind_has_a_distinct_persona(kind):
    prompt = RoleSpec("r", kind).system_prompt()
    assert prompt.startswith("You are the ")
    assert f"the {kind} " in prompt


def test_personas_differ_between_kinds():
    prompts = {RoleSpec("r", k).system_prompt() for k in ("planner", "coder", "tester", "reviewer")}
    assert len(prompts) == 4


# --- diff_genomes --------------------------------------------------------------------


def test_identical_genomes_have_empty_diff():
    seed = seed_genome()
    assert diff_genomes(seed, seed) == []
    assert diff_genomes(seed, OrgGenome.from_dict(seed.to_dict())) == []


def test_diff_ignores_lineage_fields():
    seed = seed_genome()
    child = OrgGenome(roles=seed.roles, max_retries=1, generation=3, mutation_note="x")
    assert diff_genomes(seed, child) == []


def test_diff_added_role():
    seed = seed_genome()
    grown = OrgGenome(roles=(*seed.roles, RoleSpec("reviewer", "reviewer", (), "small")))
    assert diff_genomes(seed, grown) == ["+ role reviewer (reviewer, small)"]


def test_diff_removed_role():
    seed = seed_genome()
    grown = OrgGenome(roles=(*seed.roles, RoleSpec("reviewer", "reviewer", (), "large")))
    assert diff_genomes(grown, seed) == ["- role reviewer (reviewer, large)"]


def test_diff_added_instruction():
    seed = seed_genome()
    edited = OrgGenome(
        roles=(
            RoleSpec(
                "coder",
                "coder",
                (*seed.roles[0].instructions, "Handle empty input explicitly."),
                "small",
            ),
        ),
        max_retries=1,
    )
    assert diff_genomes(seed, edited) == [
        "coder: + instruction 'Handle empty input explicitly.'"
    ]


def test_diff_removed_instruction():
    seed = seed_genome()
    edited = OrgGenome(roles=(coder(instructions=()),), max_retries=1)
    assert diff_genomes(seed, edited) == [
        "coder: - instruction 'Write a correct Python function.'"
    ]


def test_diff_replaced_instruction_shows_both_sides():
    a = org(coder(instructions=("old",)))
    b = org(coder(instructions=("new",)))
    assert diff_genomes(a, b) == ["coder: + instruction 'new'", "coder: - instruction 'old'"]


def test_diff_model_change():
    seed = seed_genome()
    large = OrgGenome(roles=(coder(model="large", instructions=seed.roles[0].instructions),))
    assert diff_genomes(seed, large) == ["coder: model small -> large"]


def test_diff_retries_change():
    seed = seed_genome()
    more = OrgGenome(roles=seed.roles, max_retries=2)
    assert diff_genomes(seed, more) == ["max_retries 1 -> 2"]


def test_diff_kind_change_for_same_role_name():
    a = org(coder(), RoleSpec("helper", "planner"))
    b = org(coder(), RoleSpec("helper", "reviewer"))
    assert diff_genomes(a, b) == ["helper: kind planner -> reviewer"]


def test_diff_role_order_does_not_matter_for_matching():
    a = org(coder(), RoleSpec("planner", "planner"))
    b = org(RoleSpec("planner", "planner"), coder())
    assert diff_genomes(a, b) == []


def test_diff_reports_several_changes_in_order():
    a = seed_genome()
    b = OrgGenome(
        roles=(
            RoleSpec(
                "coder",
                "coder",
                ("Write a correct Python function.", "Use type hints."),
                "large",
            ),
            RoleSpec("reviewer", "reviewer", (), "small"),
        ),
        max_retries=2,
    )
    assert diff_genomes(a, b) == [
        "+ role reviewer (reviewer, small)",
        "coder: model small -> large",
        "coder: + instruction 'Use type hints.'",
        "max_retries 1 -> 2",
    ]
