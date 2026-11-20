"""Shadow-execution adapter: wires ArbiterGate to a REAL deployment constraint.

Phase 0's synthetic benchmark (bench/synthetic_agent_task.py) handed the gate
an omniscient advantage signal — the harness simply knew the true delayed
benefit of both paths on every decision. That is not available in a real
deployment: you only learn the counterfactual outcome of the path you did
NOT take when you actually spend the compute to run it too (a shadow/paired
execution), and in production you cannot afford to always run both.

This adapter enforces that constraint explicitly:
  - On "exploring" / "probe" decisions, BOTH the cheap and expensive paths
    are actually executed, and the real measured advantage is reported.
  - On "trusted" / "blocked" decisions, only the chosen path is executed.
    No outcome is reported, because none was actually measured — this is
    the honest, no-cheating behavior ArbiterGate.report_outcome already
    expects (it only accepts a caller-measured advantage; it never invents
    one).

The caller supplies `cheap_fn` and `expensive_fn`, each returning a
`(deployed_payload, quality)` pair however it defines quality for its own
domain. Outcome timing is left to the caller: `decide_and_run` returns the
decision plus whatever was actually computed; recording the delayed outcome
(via `report_delayed`) can happen immediately or arbitrarily later, which is
exactly the point of ArbiterGate's ticket/callback design.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

from ..core.gate import ArbiterGate, GateDecision

T = TypeVar("T")


@dataclass
class ShadowRunResult(Generic[T]):
    decision: GateDecision
    deployed_payload: T
    deployed_cost: float
    # Only set when the decision forced a real shadow run (exploring/probe).
    # None on trusted/blocked decisions: the untaken path genuinely was not
    # executed, so its outcome is genuinely unknown, not merely withheld.
    shadow_payload: Optional[T] = None
    shadow_cost: float = 0.0
    immediate_advantage: Optional[float] = None  # measured now, if shadow ran


class ShadowExecutionAdapter(Generic[T]):
    """Pairs an ArbiterGate with real cheap_fn/expensive_fn callables and
    enforces that a counterfactual is only known when it was actually run."""

    def __init__(
        self,
        gate: ArbiterGate,
        cheap_fn: Callable[[], tuple[T, float]],
        expensive_fn: Callable[[], tuple[T, float]],
        cheap_cost: float = 1.0,
        expensive_cost: float = 1.0,
    ) -> None:
        self.gate = gate
        self.cheap_fn = cheap_fn
        self.expensive_fn = expensive_fn
        self.cheap_cost = cheap_cost
        self.expensive_cost = expensive_cost

    def decide_and_run(self, bucket: str) -> ShadowRunResult[T]:
        decision = self.gate.decide(bucket)
        run_both = decision.reason in ("exploring", "probe")

        if decision.use_expensive:
            deployed_payload, deployed_quality = self.expensive_fn()
            deployed_cost = self.expensive_cost
        else:
            deployed_payload, deployed_quality = self.cheap_fn()
            deployed_cost = self.cheap_cost

        if not run_both:
            # Honest: we did not spend the compute to learn the counterfactual.
            self.gate.abandon_outcome(decision)
            return ShadowRunResult(
                decision=decision,
                deployed_payload=deployed_payload,
                deployed_cost=deployed_cost,
            )

        if decision.use_expensive:
            shadow_payload, shadow_quality = self.cheap_fn()
            shadow_cost = self.cheap_cost
            advantage = deployed_quality - shadow_quality  # expensive - cheap
        else:
            # decide() only hands out use_expensive=False on "blocked"
            # (never "exploring"/"probe"), so run_both+not-use_expensive
            # should not occur with the current ArbiterGate — handled anyway.
            shadow_payload, shadow_quality = self.expensive_fn()
            shadow_cost = self.expensive_cost
            advantage = shadow_quality - deployed_quality

        return ShadowRunResult(
            decision=decision,
            deployed_payload=deployed_payload,
            deployed_cost=deployed_cost,
            shadow_payload=shadow_payload,
            shadow_cost=shadow_cost,
            immediate_advantage=advantage,
        )

    def report_delayed(self, decision: GateDecision, advantage: float) -> None:
        """Report a real, delayed-measured advantage for a decision that was
        run in shadow mode. Call this whenever the true outcome becomes
        known — immediately, or arbitrarily later (see gate.py docstring)."""
        self.gate.report_outcome(decision, advantage)
