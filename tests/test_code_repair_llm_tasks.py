"""Sanity checks for the Phase 3 task bank: every task must be genuinely
well-formed (correct candidate passes all hidden tests, every wrong
candidate fails at least one), and the correct answer's LETTER must not be
positionally predictable -- both caught empirically during development
(see code_repair_llm_tasks.py's module docstring and _assign_labels), now
locked in as regression tests rather than left as one-off manual checks."""
from collections import Counter

from arbiter_gate.bench.code_repair_llm_tasks import make_task_bank, run_hidden_tests, run_public_tests


def test_every_task_is_well_formed_across_several_seeds():
    for seed in (0, 1, 2, 10, 11, 12, 13, 14):
        tasks = make_task_bank(seed=seed, n_per_bucket=10)
        for t in tasks:
            correct_q, detail = run_hidden_tests(t, t.correct_label)
            assert correct_q == 1.0, f"{t.task_id}: correct candidate failed hidden tests ({detail})"
            for label in t.candidates:
                if label == t.correct_label:
                    continue
                wrong_q, _ = run_hidden_tests(t, label)
                assert wrong_q < 1.0, f"{t.task_id}: wrong candidate {label} passed every hidden test"


def test_correct_label_is_not_always_the_same_letter():
    # Regression test for the bug caught before any model call was made:
    # every task's correct answer defaulted to "A", which would have let a
    # model (or a naive parse-failure fallback) score 100% with zero real
    # reasoning.
    tasks = make_task_bank(seed=0, n_per_bucket=20)
    counts = Counter(t.correct_label for t in tasks)
    assert len(counts) > 1, "correct answer letter is not varied across tasks"
    most_common_share = max(counts.values()) / len(tasks)
    assert most_common_share < 0.6, f"one letter dominates the correct answers: {counts}"


def test_boundary_and_logic_buckets_have_genuine_public_test_ambiguity():
    # The deep path's value only shows up where public tests alone can't
    # already resolve the task -- verify that's actually true for these
    # two buckets, as designed.
    tasks = make_task_bank(seed=0, n_per_bucket=10)
    for bucket in ("boundary_bug", "logic_bug"):
        bucket_tasks = [t for t in tasks if t.bucket == bucket]
        ambiguous = sum(
            1 for t in bucket_tasks
            if sum(1 for l in t.candidates if run_public_tests(t, l)[0] == 1.0) > 1
        )
        assert ambiguous == len(bucket_tasks), f"{bucket}: expected every task ambiguous under public tests alone"


def test_arith_bug_single_public_test_is_a_full_oracle_as_designed():
    # Documented, disclosed quirk: arith_bug's one public test point (x=0)
    # happens to fully discriminate all 4 candidates by construction
    # (verified, not assumed) -- different dynamic than the other two
    # buckets, not a bug.
    tasks = make_task_bank(seed=0, n_per_bucket=10)
    bucket_tasks = [t for t in tasks if t.bucket == "arith_bug"]
    for t in bucket_tasks:
        passers = [l for l in t.candidates if run_public_tests(t, l)[0] == 1.0]
        assert passers == [t.correct_label]
