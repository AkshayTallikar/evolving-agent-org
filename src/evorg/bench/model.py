"""The task record shared by every benchmark module."""

from __future__ import annotations

from dataclasses import dataclass

SPLITS = ("train", "holdout", "all")


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    prompt: str  # natural-language spec
    signature: str  # e.g. "def rle_encode(s: str) -> str:"
    visible_tests: tuple[str, ...]  # assert statements shown to the organization
    hidden_tests: tuple[str, ...]  # assert statements used only for scoring
    reference_solution: str
    buggy_solutions: tuple[str, ...]
    difficulty: float  # 0.0 (trivial) .. 3.0 (hard)
    split: str  # "train" | "holdout"
