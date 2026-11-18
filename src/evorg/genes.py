"""The gene pool: instruction snippets an organization can adopt into a role's prompt.

The pool deliberately mixes helpful, neutral and harmful instructions. Evolution has no
label telling it which is which; selection on benchmark results has to find out.
"""

INSTRUCTION_POOL: tuple[str, ...] = (
    # plausibly helpful
    "Consider edge cases such as empty input, a single element and duplicates.",
    "Work through the problem step by step before writing code.",
    "Mentally test the function against the examples before answering.",
    "Watch for off-by-one errors at loop boundaries.",
    "Handle empty input explicitly.",
    "Use type hints.",
    # neutral
    "Write a short docstring.",
    "Prefer descriptive variable names.",
    "Keep functions under 30 lines.",
    # plausibly harmful
    "Be brief and skip unnecessary checks.",
    "If unsure, guess the most likely behaviour.",
    "Optimise for speed over correctness.",
)

ROLE_KINDS: tuple[str, ...] = ("planner", "coder", "tester", "reviewer")
MODEL_TIERS: tuple[str, ...] = ("small", "large")
