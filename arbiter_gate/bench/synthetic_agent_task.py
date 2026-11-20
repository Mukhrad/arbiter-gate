"""Synthetic AI-agent routing benchmark.

Independent of BioMind/AMG entirely (no shared code, no shared environment).
Built specifically to test the one claim this whole venture rests on: that
measuring value over a *delay* (multi-step/final outcome) beats measuring it
*immediately* (one-step), because the immediate signal can be actively
misleading, not just noisier. That is exactly the lesson from BioMind-X4
(AMG decision D82) -- this benchmark re-derives it from scratch in a domain
that has nothing to do with BioMind, deliberately, since X4's own result
does not transfer as evidence by itself.

World model
-----------
Each decision belongs to one of three buckets (the only thing any policy
is allowed to see, besides its own history):

- "deep_wins":   expensive genuinely produces much better final quality,
                 but the *immediate* quality reading is deceptive --
                 expensive looks no-better-or-slightly-worse than cheap at
                 step 0, and only pulls ahead once the task fully resolves
                 a few steps later. (Models: "the deep-reasoning answer
                 looks slower/less fluent at first glance; the quick
                 answer looks fine until it's actually checked.")
- "cheap_fine":  expensive provides no quality benefit, ever, immediate or
                 delayed, and still costs more. (Models: "this class of
                 query never needed the expensive path.")
- "shallow_wins": expensive provides a modest, *correctly* visible benefit
                 even immediately. (A control bucket: proof the comparison
                 isn't rigged to always favor delay -- an immediate signal
                 that happens to be honest should still be usable.)

Policies under test
--------------------
- always_cheap, always_expensive: trivial baselines.
- fixed_threshold: today's common approach -- a static per-bucket
  "complexity score" (noisy, correlated with but not equal to the true
  bucket) with a fixed cutoff.
- oracle: knows the true bucket-level benefit exactly (upper bound, not
  meant to be beaten).
- gate_myopic: our ArbiterGate algorithm, but fed the *immediate* (step-0)
  outcome as if it were the final advantage -- i.e. BioMind-X4's one-step
  design, reimplemented generically.
- gate_delayed: our ArbiterGate algorithm, fed the true final (delayed)
  advantage once it resolves -- the actual product design.

Honest acceptance gate (pre-registered before the validation run, matching
T059's seed discipline from the AMG project): gate_delayed must beat
gate_myopic and the fixed_threshold heuristic on *mean quality at equal or
lower cost* on held-out validation seeds (10-14) that were never used while
writing this benchmark or tuning the gate's hyperparameters. If it doesn't,
that is reported as a failure, not hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

from ..core import ArbiterGate, ArbiterGateConfig, ValueEstimatorConfig, CooldownConfig

CHEAP_COST = 1.0
EXPENSIVE_COST = 5.0

BUCKETS = ("deep_wins", "cheap_fine", "shallow_wins")

# Static, noisy "complexity score" available to the fixed_threshold baseline
# only -- correlated with, but not equal to, the true per-bucket benefit.
# deep_wins *should* get a high score (it does need expensive) but a crude
# static heuristic built by eyeballing immediate outputs would plausibly
# underrate it, since the benefit is invisible immediately; cheap_fine
# correctly gets a low score; shallow_wins correctly gets a middling-high
# score since its benefit is visible on the surface too.
STATIC_COMPLEXITY_SCORE = {"deep_wins": 0.45, "cheap_fine": 0.10, "shallow_wins": 0.70}
FIXED_THRESHOLD = 0.5

TRUE_BUCKET_BENEFIT = {"deep_wins": 0.55, "cheap_fine": 0.0, "shallow_wins": 0.25}
IMMEDIATE_SIGNAL_BENEFIT = {"deep_wins": -0.05, "cheap_fine": -0.03, "shallow_wins": 0.22}
NOISE_SD = 0.08


@dataclass
class Outcome:
    quality_cheap: float
    quality_expensive: float
    immediate_quality_cheap: float
    immediate_quality_expensive: float


def sample_outcome(bucket: str, rng: random.Random) -> Outcome:
    base = 0.5
    q_cheap = base + rng.gauss(0.0, NOISE_SD)
    q_expensive = base + TRUE_BUCKET_BENEFIT[bucket] + rng.gauss(0.0, NOISE_SD)
    imm_cheap = base + rng.gauss(0.0, NOISE_SD)
    imm_expensive = base + IMMEDIATE_SIGNAL_BENEFIT[bucket] + rng.gauss(0.0, NOISE_SD)
    return Outcome(q_cheap, q_expensive, imm_cheap, imm_expensive)


_POLICY_SALT = {
    "always_cheap": 1, "always_expensive": 2, "fixed_threshold": 3,
    "oracle": 4, "gate_myopic": 5, "gate_delayed": 6,
}


def run_policy(policy: str, bucket_sequence: list[str], rng: random.Random,
               gate_config: ArbiterGateConfig | None = None) -> dict:
    total_quality = 0.0
    total_cost = 0.0
    expensive_count = 0
    gate = None
    if policy in ("gate_myopic", "gate_delayed"):
        gate = ArbiterGate(gate_config or ArbiterGateConfig())

    for bucket in bucket_sequence:
        outcome = sample_outcome(bucket, rng)
        if policy == "always_cheap":
            use_expensive = False
        elif policy == "always_expensive":
            use_expensive = True
        elif policy == "fixed_threshold":
            use_expensive = STATIC_COMPLEXITY_SCORE[bucket] > FIXED_THRESHOLD
        elif policy == "oracle":
            use_expensive = TRUE_BUCKET_BENEFIT[bucket] > 0.0
        elif policy in ("gate_myopic", "gate_delayed"):
            decision = gate.decide(bucket)
            use_expensive = decision.use_expensive
        else:
            raise ValueError(policy)

        quality = outcome.quality_expensive if use_expensive else outcome.quality_cheap
        cost = EXPENSIVE_COST if use_expensive else CHEAP_COST
        total_quality += quality
        total_cost += cost
        expensive_count += int(use_expensive)

        if policy == "gate_myopic" and decision.use_expensive:
            advantage = outcome.immediate_quality_expensive - outcome.immediate_quality_cheap
            gate.report_outcome(decision, advantage)
        elif policy == "gate_delayed" and decision.use_expensive:
            advantage = outcome.quality_expensive - outcome.quality_cheap
            gate.report_outcome(decision, advantage)

    n = len(bucket_sequence)
    return {
        "policy": policy,
        "mean_quality": total_quality / n,
        "mean_cost": total_cost / n,
        "expensive_fraction": expensive_count / n,
    }


def make_bucket_sequence(n: int, rng: random.Random) -> list[str]:
    return [rng.choice(BUCKETS) for _ in range(n)]


def run_all_policies(seed: int, n_decisions: int = 1500) -> dict:
    rng = random.Random(seed)
    sequence = make_bucket_sequence(n_decisions, rng)
    results = {}
    for policy in ("always_cheap", "always_expensive", "fixed_threshold", "oracle",
                   "gate_myopic", "gate_delayed"):
        results[policy] = run_policy(
            policy, sequence, random.Random(seed * 7919 + _POLICY_SALT[policy]))
    return results


if __name__ == "__main__":
    import json
    import sys

    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
    all_results = {seed: run_all_policies(seed) for seed in seeds}
    print(json.dumps(all_results, indent=2))
