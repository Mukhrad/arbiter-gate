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

    STATE_SCHEMA_VERSION = 1

    def export_state(self) -> dict:
        """Serializable snapshot of everything this gate has LEARNED so far:
        the exploration counter per bucket, the settled value-estimator
        statistics, and the cooldown state. Deliberately excludes any open
        (not yet reported/abandoned) decision tickets -- see
        MultiStepValueEstimator.export_stats for why. A caller that persists
        this and reloads it into a fresh ArbiterGate (e.g. a new CI run, a
        new process) gets a gate whose very first decision already reflects
        every prior run's evidence, instead of re-exploring from zero."""
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "decisions_per_bucket": dict(self.decisions_per_bucket),
            "value_stats": self.value.export_stats(),
            "cooldown_state": self.cooldown.export_state(),
        }

    def load_state(self, state: dict) -> None:
        """Restore a snapshot produced by `export_state`. All-or-nothing: every
        field (schema_version, decisions_per_bucket, value_stats,
        cooldown_state) is fully parsed and validated BEFORE anything on this
        gate is mutated, so a failure anywhere -- a bad schema version, a
        malformed bucket count, a non-finite or negative statistic -- leaves
        this gate exactly as it was before the call, never partially
        applied. (An earlier version of this method validated and mutated
        decisions_per_bucket/value/cooldown one at a time, which could leave
        decisions_per_bucket advanced past the exploration threshold while
        value_stats/cooldown_state silently failed to load -- causing the
        very next decision to skip exploration with no real evidence behind
        it. Found and fixed after an independent reviewer reproduced it.)"""
        version = state.get("schema_version")
        if version != self.STATE_SCHEMA_VERSION:
            raise ValueError(
                f"ArbiterGate state schema_version {version!r} is not "
                f"supported by this library version (expects "
                f"{self.STATE_SCHEMA_VERSION!r}); refusing to load it rather "
                f"than guess at its meaning."
            )

        raw_decisions = state.get("decisions_per_bucket", {})
        decisions_per_bucket: dict[str, int] = {}
        for bucket, n in raw_decisions.items():
            n = int(n)
            if n < 0:
                raise ValueError(f"decisions_per_bucket[{bucket!r}] must be >= 0, got {n!r}")
            decisions_per_bucket[bucket] = n

        # Validate value/cooldown state too, still without touching self --
        # a failure in either of these leaves decisions_per_bucket (above)
        # un-applied as well.
        staged_value = self.value.validate_export(state.get("value_stats", {}))
        staged_cooldown = self.cooldown.validate_export(state.get("cooldown_state", {}))

        # Every field validated; commit all three together. Each of these
        # three calls is a plain merge of already-validated data and cannot
        # itself fail, so this is effectively one atomic commit.
        self.decisions_per_bucket.update(decisions_per_bucket)
        self.value.apply_staged(staged_value)
        self.cooldown.apply_staged(staged_cooldown)

    def diagnostics(self) -> dict:
        return {
            "value": self.value.diagnostics(),
            "cooldown": self.cooldown.diagnostics(),
            "decisions_per_bucket": dict(self.decisions_per_bucket),
        }
