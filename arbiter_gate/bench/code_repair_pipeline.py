"""Phase 1 demo: a REAL, executed repair-decision pipeline with genuinely
delayed consequences — the thing Phase 0's VALIDATION_REPORT explicitly
flagged as unproven ("advantage is handed by an omniscient benchmark
harness, not recovered from real paired/shadow execution").

Scenario (deliberately simplified but not faked): an agent repairs small
broken helper functions as they're introduced into a pipeline. Each helper
has some number of hidden boolean "repair parameters" (how to patch it) with
exactly one correct combination, unknown to the fixer. Two repair strategies
are REAL, EXECUTED code — not numbers pulled from a distribution:

  - cheap_fix:  returns a fixed default guess. No testing at all. O(1).
  - expensive_fix: actually tries every parameter combination against the
    small set of "smoke" tests available at commit time, and keeps the
    first one that passes all of them. O(2**n_params).

The twist that makes delayed credit assignment matter for real: the smoke
tests available *at commit time* are few, and for buckets with more than one
hidden parameter they typically under-determine the correct combination
(multiple combos pass smoke). The *true* correctness only shows up against a
larger "edge" test set that is only exercised `offset` pipeline-steps later
(mirrors a regression surfacing several commits downstream in real CI). This
is genuinely executed Python comparing genuinely computed outputs — nothing
here is sampled from a hand-picked probability table.

Buckets (chosen so the "value of being thorough" genuinely differs, as a
emergent property of the arithmetic below, not an authored label):
  - "offbyone_bug"  (n_params=1): 2 combos; 2 smoke points fully pin it down
     -> expensive is nearly always exactly right; cheap is right ~50% by luck.
  - "comparison_bug" (n_params=2): 4 combos; 2 smoke points under-determine
     it about half the time -> expensive helps a lot but not perfectly.
  - "arithmetic_bug" (n_params=3): 8 combos; 2 smoke points badly
     under-determine it -> expensive still helps (more candidates checked)
     but with real, measured uncertainty; cheap is right ~1/8 by luck.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import random
import statistics
from typing import Optional

from ..adapters.shadow import ShadowExecutionAdapter
from ..core.gate import ArbiterGate, ArbiterGateConfig, GateDecision
from ..core.value_estimator import ValueEstimatorConfig

BUCKET_N_PARAMS = {
    "offbyone_bug": 1,
    "comparison_bug": 2,
    "arithmetic_bug": 3,
    "trivial_bug": 1,
}
# Probability the bug's correct repair happens to BE the cheap default guess
# for that bucket. 0.0 means the correct combo is uniformly random (cheap
# is right only by luck); "trivial_bug" is deliberately biased so the quick
# default patch is usually already correct — the case a good gate should
# learn to stop paying for expensive on, not just "always use expensive".
BUCKET_DEFAULT_BIAS = {
    "offbyone_bug": 0.0,
    "comparison_bug": 0.0,
    "arithmetic_bug": 0.0,
    "trivial_bug": 0.85,
}
BUCKETS = tuple(BUCKET_N_PARAMS.keys())
MODULUS = 11  # small, arbitrary; just needs to make the linear system nontrivial

EXPENSIVE_COST = {b: float(2 ** n) for b, n in BUCKET_N_PARAMS.items()}
CHEAP_COST = 1.0


def _base_value(x: int) -> int:
    return (x * 3 + 1) % MODULUS


def _weight(i: int, x: int) -> int:
    # Deterministic but non-linear/non-separable in (i, x), via hashlib
    # (NOT Python's hash(), which is randomized per-process by default —
    # the exact bug disclosed in Phase 0's VALIDATION_REPORT). A linear
    # weight(i, x) made the two smoke points algebraically redundant
    # (x=2's equation was just 2x x=1's), which accidentally let the smoke
    # tests always fully determine the correct combo — defeating the point
    # of this benchmark. Hashing the (i, x) pair avoids that redundancy, so
    # smoke genuinely under-determines the combo for n_params >= 2, as
    # intended.
    digest = hashlib.sha256(f"{i}:{x}".encode()).digest()
    return digest[0] % MODULUS


def _fn(combo: tuple[int, ...], x: int) -> int:
    return (_base_value(x) + sum(c * _weight(i, x) for i, c in enumerate(combo))) % MODULUS


@dataclass(frozen=True)
class RepairInstance:
    bucket: str
    n_params: int
    correct_combo: tuple[int, ...]
    default_combo: tuple[int, ...]
    smoke_inputs: tuple[int, ...]
    edge_inputs: tuple[int, ...]

    def true_value(self, x: int) -> int:
        return _fn(self.correct_combo, x)

    def quality(self, combo: tuple[int, ...], inputs: tuple[int, ...]) -> float:
        if not inputs:
            return 0.0
        hits = sum(1 for x in inputs if _fn(combo, x) == self.true_value(x))
        return hits / len(inputs)


def make_instance(bucket: str, rng: random.Random) -> RepairInstance:
    n = BUCKET_N_PARAMS[bucket]
    default = tuple(0 for _ in range(n))  # cheap's blind, untested guess
    if rng.random() < BUCKET_DEFAULT_BIAS[bucket]:
        correct = default
    else:
        correct = tuple(rng.randint(0, 1) for _ in range(n))
    smoke = (1, 2)  # small, fixed smoke set available at commit time
    edge = tuple(range(1, 9))  # larger set only exercised later
    return RepairInstance(
        bucket=bucket, n_params=n, correct_combo=correct, default_combo=default,
        smoke_inputs=smoke, edge_inputs=edge,
    )


def cheap_fix(instance: RepairInstance) -> tuple[tuple[int, ...], float]:
    """Blind default guess, no execution-based verification at all."""
    combo = instance.default_combo
    return combo, instance.quality(combo, instance.edge_inputs)


def expensive_fix(instance: RepairInstance) -> tuple[tuple[int, ...], float]:
    """Actually try every combo, actually run it against the smoke tests,
    keep the first that actually passes all of them."""
    for combo in itertools.product((0, 1), repeat=instance.n_params):
        if all(_fn(combo, x) == instance.true_value(x) for x in instance.smoke_inputs):
            return combo, instance.quality(combo, instance.edge_inputs)
    # No combo passed smoke (shouldn't happen since correct_combo always
    # does) — fall back to the default guess.
    combo = instance.default_combo
    return combo, instance.quality(combo, instance.edge_inputs)


@dataclass
class _OpenTicket:
    decision: GateDecision
    immediate_advantage: Optional[float]
    due_at_step: int


def run_pipeline(seed: int, n_decisions: int = 1200, offset: int = 3,
                  gate_config: Optional[ArbiterGateConfig] = None) -> dict:
    """Runs four policies over the SAME sequence of real repair instances:
    always_cheap, always_expensive, static_rule (hand-written, bucket-based),
    and the gate (shadow-executed, delayed-reported). Returns per-policy
    mean deployed quality (the real edge-test pass rate actually shipped)
    and mean deployed cost.
    """
    rng = random.Random(seed * 104729)
    bucket_sequence = [BUCKETS[i % len(BUCKETS)] for i in range(n_decisions)]
    rng.shuffle(bucket_sequence)
    instances = [make_instance(b, rng) for b in bucket_sequence]

    results: dict[str, dict] = {}

    def record(name: str, qualities: list[float], costs: list[float]) -> None:
        results[name] = {
            "quality_mean": statistics.fmean(qualities),
            "quality_sd": statistics.pstdev(qualities) if len(qualities) > 1 else 0.0,
            "cost_mean": statistics.fmean(costs),
            "n": len(qualities),
        }

    # --- always_cheap ---
    qs, cs = [], []
    for inst in instances:
        _, q = cheap_fix(inst)
        qs.append(q)
        cs.append(CHEAP_COST)
    record("always_cheap", qs, cs)

    # --- always_expensive ---
    qs, cs = [], []
    for inst in instances:
        _, q = expensive_fix(inst)
        qs.append(q)
        cs.append(EXPENSIVE_COST[inst.bucket])
    record("always_expensive", qs, cs)

    # --- static_rule: a reasonable hand-written policy an engineer might
    # ship without any learning mechanism — expensive only for the bucket
    # that "sounds" hardest (most parameters), cheap otherwise.
    qs, cs = [], []
    for inst in instances:
        if inst.bucket in ("arithmetic_bug", "comparison_bug"):
            _, q = expensive_fix(inst)
            c = EXPENSIVE_COST[inst.bucket]
        else:
            _, q = cheap_fix(inst)
            c = CHEAP_COST
        qs.append(q)
        cs.append(c)
    record("static_rule", qs, cs)

    # --- gate_shadow: real shadow execution, real delayed reporting ---
    gate = ArbiterGate(gate_config or ArbiterGateConfig())
    deployed_q, deployed_c, shadow_c = [], [], []
    pending: list[_OpenTicket] = []

    for step, inst in enumerate(instances):
        adapter = ShadowExecutionAdapter(
            gate=gate,
            cheap_fn=lambda inst=inst: cheap_fix(inst),
            expensive_fn=lambda inst=inst: expensive_fix(inst),
            cheap_cost=CHEAP_COST,
            expensive_cost=EXPENSIVE_COST[inst.bucket],
        )
        run = adapter.decide_and_run(inst.bucket)
        # Quality actually deployed is the edge-test quality of the combo we
        # shipped — recomputed directly from the payload (the chosen combo).
        deployed_q.append(inst.quality(run.deployed_payload, inst.edge_inputs))
        deployed_c.append(run.deployed_cost)
        shadow_c.append(run.shadow_cost)
        if run.immediate_advantage is not None:
            pending.append(_OpenTicket(
                decision=run.decision,
                immediate_advantage=run.immediate_advantage,
                due_at_step=step + offset,
            ))
        # Resolve any tickets whose delay has elapsed: report the REAL,
        # already-measured advantage now that "enough pipeline steps" have
        # passed (mirrors a regression surfacing several commits later).
        still_pending = []
        for t in pending:
            if step >= t.due_at_step:
                adapter.report_delayed(t.decision, t.immediate_advantage)
            else:
                still_pending.append(t)
        pending = still_pending

    # Flush anything still open at the end (would otherwise just expire).
    for t in pending:
        adapter.report_delayed(t.decision, t.immediate_advantage)

    record("gate_shadow", deployed_q, deployed_c)
    results["gate_shadow"]["shadow_cost_mean"] = statistics.fmean(shadow_c)
    results["gate_shadow"]["diagnostics"] = gate.diagnostics()

    return results


def run_all_seeds(seeds: list[int], n_decisions: int = 1200, offset: int = 3) -> dict:
    # Phase 1 pre-registered this margin (value_margin=0.05, see
    # PHASE1_REPORT.md) BEFORE touching the reserved validation seeds.
    # Caught by independent review: this entry point had silently drifted
    # to the library default (0.0) instead, so the documented reproduction
    # command (`python -m arbiter_gate.bench.code_repair_pipeline 10 11 12
    # 13 14`) did not actually reproduce phase1_validation_results.json.
    # Pinned here so the public command matches the pre-registered protocol.
    config = ArbiterGateConfig(value_config=ValueEstimatorConfig(value_margin=0.05))
    return {
        seed: run_pipeline(seed, n_decisions=n_decisions, offset=offset, gate_config=config)
        for seed in seeds
    }


if __name__ == "__main__":
    import json
    import sys

    seeds = [int(s) for s in sys.argv[1:]] or [10, 11, 12, 13, 14]
    print(json.dumps(run_all_seeds(seeds), indent=2, default=str))
