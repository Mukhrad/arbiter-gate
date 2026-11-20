"""The public arbiter gate: decide cheap-vs-expensive per context, learn from
delayed/multi-step outcomes, revert automatically on measured harm.

Usage sketch:

    gate = ArbiterGate()
    decision = gate.decide(bucket="code_question", context=ctx)
    if decision.use_expensive:
        result = expensive_fn(ctx)
    else:
        result = cheap_fn(ctx)
    ... later, once the real outcome is known (could be the same turn or
    many turns later) ...
    gate.report_outcome(decision, advantage=measured_expensive_minus_cheap)

`advantage` is deliberately the caller's responsibility to compute: this
library does not assume it can rewind reality to get a true counterfactual.
The caller may use a true simulator rollback (best, when available), a
paired/shadow execution sampled on a fraction of decisions, or a
regression-adjusted estimate. See bench/synthetic_agent_task.py for a worked
example with paired shadow sampling.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cooldown import CooldownConfig, CooldownController
from .value_estimator import MultiStepValueEstimator, ValueEstimatorConfig


@dataclass(frozen=True)
class GateDecision:
    decision_id: str
    bucket: str
    use_expensive: bool
    reason: str  # "exploring" | "trusted" | "blocked" | "probe"


@dataclass(frozen=True)
class ArbiterGateConfig:
    exploratory_decisions: int = 4   # per bucket, before we trust any estimate
    value_config: ValueEstimatorConfig = None  # type: ignore[assignment]
    cooldown_config: CooldownConfig = None      # type: ignore[assignment]

    def __post_init__(self):
        object.__setattr__(self, "value_config", self.value_config or ValueEstimatorConfig())
        object.__setattr__(self, "cooldown_config", self.cooldown_config or CooldownConfig())


class ArbiterGate:
    def __init__(self, config: ArbiterGateConfig | None = None) -> None:
        self.config = config or ArbiterGateConfig()
        self.value = MultiStepValueEstimator(self.config.value_config)
        self.cooldown = CooldownController(self.config.cooldown_config)
        self.decisions_per_bucket: dict[str, int] = {}

    def decide(self, bucket: str) -> GateDecision:
        n = self.decisions_per_bucket.get(bucket, 0)
        self.decisions_per_bucket[bucket] = n + 1

        if n < self.config.exploratory_decisions:
            decision_id = self.value.open_decision(bucket, took_expensive=True)
            return GateDecision(decision_id, bucket, True, "exploring")

        lcb = self.value.value_lcb(bucket)
        if lcb is not None and lcb > self.config.value_config.value_margin:
            self.cooldown.note_allowed(bucket)
            decision_id = self.value.open_decision(bucket, took_expensive=True)
            return GateDecision(decision_id, bucket, True, "trusted")

        if self.cooldown.allow_despite_block(bucket):
            decision_id = self.value.open_decision(bucket, took_expensive=True)
            return GateDecision(decision_id, bucket, True, "probe")

        # Blocked: take the cheap path. Still open a (took_expensive=False)
        # decision purely so report_outcome has a consistent id to accept
        # and no-op on, keeping the caller's code path uniform.
        decision_id = self.value.open_decision(bucket, took_expensive=False)
        return GateDecision(decision_id, bucket, False, "blocked")

    def report_outcome(self, decision: GateDecision, advantage: float) -> None:
        """`advantage` = realized_outcome(expensive) - realized_outcome(cheap)
        for this decision, however the caller measured or estimated it.
        No-op for decisions that took the cheap path (see value_estimator)."""
        self.value.record_outcome(decision.decision_id, advantage)
        if decision.use_expensive:
            self.cooldown.note_realized_value(
                decision.bucket, advantage, self.config.value_config.value_margin)

    def abandon_outcome(self, decision: GateDecision) -> bool:
        """Close a decision when the caller did not run a counterfactual."""
        return self.value.abandon_decision(decision.decision_id)

    def diagnostics(self) -> dict:
        return {
            "value": self.value.diagnostics(),
            "cooldown": self.cooldown.diagnostics(),
            "decisions_per_bucket": dict(self.decisions_per_bucket),
        }
