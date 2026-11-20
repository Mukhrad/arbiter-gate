"""Pre-registered harder Phase-4 task presentation.

The executable candidates and hidden judges remain the verified Phase-3
tasks.  Difficulty changes only what the quick path is told.  A hard item
withholds the decisive natural-language convention from quick, while deep
receives the full specification plus public-test evidence.  Difficulty is
mixed *within* each family, with fixed rates chosen before validation:
boundary=.80, arithmetic=.50, logic=.20.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

from .code_repair_llm_tasks import RepairTask, make_task_bank


HARD_RATES = {
    "boundary_bug": 0.80,
    "arith_bug": 0.50,
    "logic_bug": 0.20,
}


@dataclass(frozen=True)
class Phase4Task:
    base: RepairTask
    quick_spec_text: str
    difficulty: str

    @property
    def task_id(self): return self.base.task_id
    @property
    def bucket(self): return self.base.bucket
    @property
    def spec_text(self): return self.base.spec_text
    @property
    def candidates(self): return self.base.candidates
    @property
    def correct_label(self): return self.base.correct_label
    @property
    def func_name(self): return self.base.func_name
    @property
    def public_tests(self): return self.base.public_tests
    @property
    def hidden_tests(self): return self.base.hidden_tests


def _redacted_spec(task: RepairTask) -> str:
    if task.bucket == "boundary_bug":
        return (
            f"Select the intended implementation of `{task.func_name}`. "
            "The short request does not specify whether either endpoint is inclusive."
        )
    if task.bucket == "arith_bug":
        return (
            f"Select the intended implementation of `{task.func_name}`. "
            "The short request says only that it is a scale/shift helper; the exact formula is omitted."
        )
    return (
        f"Select the intended implementation of `{task.func_name}`. "
        "The short request omits the exact eligibility conjunction and boundary convention."
    )


def make_phase4_task_bank(seed: int, n_per_bucket: int = 10) -> list[Phase4Task]:
    base = make_task_bank(seed=seed, n_per_bucket=n_per_bucket)
    rng = random.Random(seed * 104729 + 20261119)
    out = []
    for task in base:
        hard = rng.random() < HARD_RATES[task.bucket]
        out.append(Phase4Task(
            base=task,
            quick_spec_text=_redacted_spec(task) if hard else task.spec_text,
            difficulty="hard" if hard else "clear",
        ))
    return out
