"""Phase 3 task bank: real, hand-verified Python bug-repair tasks for the
Ollama/qwen2.5:7b experiment.

SAFETY DESIGN (explicit, per the founder's instruction): the model never
generates or runs free-form code. For every task it is shown a FIXED,
pre-written set of 3-4 candidate full implementations (labelled A, B, C, ...)
and asked only to name the letter it believes is correct. We apply the
candidate's PRE-WRITTEN source (never anything the model typed) in an
isolated temp directory, in a subprocess, with a timeout. The model's raw
output is parsed only for a single letter; anything else is logged and
treated as "no answer" (falls back to a fixed default, never executed
blindly as code).

Three bucket types, each a parameterized family (so many real, distinct
task instances come from one hand-verified template + per-instance random
constants) — same spirit as Phase 1's offbyone/comparison/arithmetic split,
but now the "candidates" are things a real small LLM can plausibly reason
about from a natural-language spec, not just enumerable boolean vectors:

  - "boundary_bug":  inclusive-range check, candidates differ in which
    comparison operators (< vs <=) are used at each bound.
  - "arith_bug":     linear transform x*a+b, candidates differ in which
    operator combination is used.
  - "logic_bug":     boolean eligibility rule, candidates differ in which
    boolean operator / comparison / negation is used.

Public tests (shown to the deep path) are deliberately few and sometimes
under-discriminating, mirroring Phase 1's smoke/edge split; hidden tests
(used only for final grading, never shown to the model) are more numerous
and chosen to actually separate the candidates. This is verified
empirically for each bucket in `arbiter_gate/tests/test_code_repair_llm_tasks.py`,
not just asserted in a docstring.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

TEST_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class RepairTask:
    task_id: str
    bucket: str
    spec_text: str          # natural-language spec shown to the model
    candidates: dict        # label -> full function source (str)
    correct_label: str      # ground truth, never shown to the model
    func_name: str
    public_tests: list      # list of (call_args_repr, expected_repr) strings
    hidden_tests: list      # same shape, used only for grading


def _run_candidate(source: str, func_name: str, test_lines: list) -> tuple[float, str]:
    """Writes `source` (one of our own fixed candidates — never model-typed
    code) to an isolated temp file and runs the given assert-style test
    lines against it in a subprocess with a timeout. Returns
    (fraction_passed, detail_text)."""
    # Built by direct concatenation, not textwrap.dedent: `source` is
    # already valid top-level Python with its own correct internal
    # indentation, and dedent()-ing a template that embeds a multi-line
    # variable mangles that inner indentation (caught empirically — see
    # test_code_repair_llm_tasks.py — not just asserted).
    harness_lines = [
        source,
        "",
        "_results = []",
        *test_lines,
        "import json",
        "print(json.dumps(_results))",
    ]
    harness = "\n".join(harness_lines)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "candidate.py"
        path.write_text(harness, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)],
                capture_output=True, text=True, timeout=TEST_TIMEOUT_SECONDS,
                cwd=td,
            )
        except subprocess.TimeoutExpired:
            return 0.0, "TIMEOUT"
        if proc.returncode != 0:
            return 0.0, f"ERROR: {proc.stderr.strip()[-500:]}"
        import json
        try:
            results = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return 0.0, f"UNPARSEABLE_OUTPUT: {proc.stdout[-500:]}"
        passed = sum(1 for r in results if r)
        return (passed / len(results) if results else 0.0), f"{passed}/{len(results)} passed"


def run_public_tests(task: RepairTask, label: str) -> tuple[float, str]:
    source = task.candidates[label]
    lines = [f"_results.append({call} == {expected})" for call, expected in task.public_tests]
    return _run_candidate(source, task.func_name, lines)


def run_hidden_tests(task: RepairTask, label: str) -> tuple[float, str]:
    source = task.candidates[label]
    lines = [f"_results.append({call} == {expected})" for call, expected in task.hidden_tests]
    return _run_candidate(source, task.func_name, lines)


_LABELS = ["A", "B", "C", "D"]


def _assign_labels(variants: list, rng: random.Random) -> tuple[dict, str]:
    """`variants[0]` is always the correct implementation by construction
    above; this shuffles which LETTER it ends up labelled as, per instance.
    Without this, every task's correct answer would be "A" — a model (or
    even just the parser's fallback-on-failure path) could score 100% by
    always answering "A" with zero real reasoning, which would silently
    invalidate the whole experiment. Caught before any model call was
    made, not after looking at results."""
    order = list(range(len(variants)))
    rng.shuffle(order)
    candidates = {_LABELS[i]: variants[order[i]] for i in range(len(variants))}
    correct_label = _LABELS[order.index(0)]
    return candidates, correct_label


# --------------------------------------------------------------- boundary_bug

def _make_boundary_bug(rng: random.Random, idx: int) -> RepairTask:
    lo = rng.randint(-20, 10)
    hi = lo + rng.randint(5, 25)
    fn = f"in_range_{idx}"
    variants = [
        f"def {fn}(x, lo={lo}, hi={hi}):\n    return lo <= x <= hi",  # correct
        f"def {fn}(x, lo={lo}, hi={hi}):\n    return lo < x < hi",
        f"def {fn}(x, lo={lo}, hi={hi}):\n    return lo <= x < hi",
        f"def {fn}(x, lo={lo}, hi={hi}):\n    return lo < x <= hi",
    ]
    candidates, correct_label = _assign_labels(variants, rng)
    mid = (lo + hi) // 2
    public = [(f"{fn}({mid})", "True"), (f"{fn}({lo - 5})", "False")]
    hidden = [
        (f"{fn}({lo})", "True"), (f"{fn}({hi})", "True"),
        (f"{fn}({lo - 1})", "False"), (f"{fn}({hi + 1})", "False"),
    ]
    spec = (
        f"Write/identify a function `{fn}(x, lo={lo}, hi={hi})` that returns True "
        f"if and only if x is between lo and hi, INCLUSIVE of both endpoints "
        f"(x == lo or x == hi must both return True)."
    )
    return RepairTask(f"boundary_{idx}", "boundary_bug", spec, candidates, correct_label, fn, public, hidden)


# ------------------------------------------------------------------ arith_bug

def _make_arith_bug(rng: random.Random, idx: int) -> RepairTask:
    a = rng.randint(2, 6)
    # b == 0 degenerately collapses "x*a-b" and "x*(a+b)" onto the correct
    # answer (caught empirically, not assumed) -- exclude it so every
    # instance is a genuine, discriminating task.
    b = rng.choice([v for v in range(-10, 11) if v != 0])
    fn = f"scale_shift_{idx}"
    variants = [
        f"def {fn}(x, a={a}, b={b}):\n    return x * a + b",  # correct
        f"def {fn}(x, a={a}, b={b}):\n    return x + a + b",
        f"def {fn}(x, a={a}, b={b}):\n    return x * a - b",
        f"def {fn}(x, a={a}, b={b}):\n    return x * (a + b)",
    ]
    candidates, correct_label = _assign_labels(variants, rng)
    public = [(f"{fn}(0)", str(b))]
    hidden = [
        (f"{fn}(1)", str(a + b)), (f"{fn}(2)", str(2 * a + b)),
        (f"{fn}(-1)", str(-a + b)), (f"{fn}(3)", str(3 * a + b)),
        (f"{fn}(5)", str(5 * a + b)),
    ]
    spec = (
        f"Write/identify a function `{fn}(x, a={a}, b={b})` that returns "
        f"x scaled by a and then shifted by b, i.e. the linear transform x*a + b."
    )
    return RepairTask(f"arith_{idx}", "arith_bug", spec, candidates, correct_label, fn, public, hidden)


# ------------------------------------------------------------------ logic_bug

def _make_logic_bug(rng: random.Random, idx: int) -> RepairTask:
    k = rng.randint(16, 21)
    fn = f"is_eligible_{idx}"
    variants = [
        f"def {fn}(age, has_id, min_age={k}):\n    return age >= min_age and has_id",  # correct
        f"def {fn}(age, has_id, min_age={k}):\n    return age >= min_age or has_id",
        f"def {fn}(age, has_id, min_age={k}):\n    return age > min_age and has_id",
        f"def {fn}(age, has_id, min_age={k}):\n    return age >= min_age and not has_id",
    ]
    candidates, correct_label = _assign_labels(variants, rng)
    public = [(f"{fn}({k + 5}, True)", "True"), (f"{fn}({k - 5}, False)", "False")]
    hidden = [
        (f"{fn}({k}, True)", "True"), (f"{fn}({k}, False)", "False"),
        (f"{fn}({k - 1}, True)", "False"), (f"{fn}({k + 2}, False)", "False"),
    ]
    spec = (
        f"Write/identify a function `{fn}(age, has_id, min_age={k})` that returns True "
        f"if and only if BOTH age >= min_age AND has_id are true (eligibility requires both)."
    )
    return RepairTask(f"logic_{idx}", "logic_bug", spec, candidates, correct_label, fn, public, hidden)


BUCKET_MAKERS = {
    "boundary_bug": _make_boundary_bug,
    "arith_bug": _make_arith_bug,
    "logic_bug": _make_logic_bug,
}


def make_task_bank(seed: int, n_per_bucket: int = 20) -> list[RepairTask]:
    rng = random.Random(seed * 486187739 + 11)
    tasks: list[RepairTask] = []
    for bucket, maker in BUCKET_MAKERS.items():
        for i in range(n_per_bucket):
            tasks.append(maker(rng, i))
    order = list(range(len(tasks)))
    rng.shuffle(order)
    return [tasks[i] for i in order]
