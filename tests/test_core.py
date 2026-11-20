from __future__ import annotations

from arbiter_gate.core import (
    ArbiterGate, ArbiterGateConfig, ValueEstimatorConfig, CooldownConfig,
    MultiStepValueEstimator,
)


def test_explores_before_trusting():
    gate = ArbiterGate(ArbiterGateConfig(exploratory_decisions=3))
    reasons = [gate.decide("b").reason for _ in range(3)]
    assert reasons == ["exploring", "exploring", "exploring"]


def test_positive_delayed_outcome_grants_trust():
    gate = ArbiterGate(ArbiterGateConfig(
        exploratory_decisions=2,
        value_config=ValueEstimatorConfig(min_samples_for_lcb=2, uncertainty_penalty=0.5),
    ))
    decisions = [gate.decide("b") for _ in range(2)]
    for d in decisions:
        gate.report_outcome(d, advantage=0.5)
    d = gate.decide("b")
    assert d.reason == "trusted"
    assert d.use_expensive


def test_negative_delayed_outcome_blocks_and_cools_down():
    gate = ArbiterGate(ArbiterGateConfig(
        exploratory_decisions=2,
        value_config=ValueEstimatorConfig(min_samples_for_lcb=2, uncertainty_penalty=0.5),
        cooldown_config=CooldownConfig(negative_streak_for_cooldown=1, cooldown_decisions=3, probe_interval=100),
    ))
    decisions = [gate.decide("b") for _ in range(2)]
    for d in decisions:
        gate.report_outcome(d, advantage=-0.5)
    # negative streak of 1 (>= threshold 1) should already trigger cooldown
    blocked = gate.decide("b")
    assert blocked.reason == "blocked"
    assert not blocked.use_expensive


def test_sparse_probe_eventually_fires_during_long_block():
    gate = ArbiterGate(ArbiterGateConfig(
        exploratory_decisions=1,
        value_config=ValueEstimatorConfig(min_samples_for_lcb=1, uncertainty_penalty=0.0, value_margin=10.0),
        cooldown_config=CooldownConfig(negative_streak_for_cooldown=100, cooldown_decisions=0, probe_interval=5),
    ))
    d0 = gate.decide("b")
    gate.report_outcome(d0, advantage=-1.0)  # measured but value_margin=10 keeps it "blocked" forever
    reasons = [gate.decide("b").reason for _ in range(12)]
    assert reasons.count("probe") >= 2  # fires roughly every 5th block


def test_decision_reported_only_once_is_safe():
    """Double-reporting or reporting a cheap-path decision must not corrupt
    stats (this was a real bug-class risk: BioMind-X4's run_episode_x4 relied
    on careful ordering around `planner.last_intervened`)."""
    est = MultiStepValueEstimator(ValueEstimatorConfig(min_samples_for_lcb=1))
    cheap_id = est.open_decision("b", took_expensive=False)
    est.record_outcome(cheap_id, advantage=999.0)
    assert est.sample_count("b") == 0  # cheap-path outcomes are never recorded as advantage evidence

    exp_id = est.open_decision("b", took_expensive=True)
    est.record_outcome(exp_id, advantage=1.0)
    est.record_outcome(exp_id, advantage=1.0)  # second report on same (now-popped) id: no-op
    assert est.sample_count("b") == 1


def test_bucket_isolation():
    """A mechanism that helps on one context and hurts on another must not
    average to 'roughly zero' globally -- this is exactly the failure mode
    that made BioMind-X4's single global-ish value signal uninformative on
    4 of 5 seeds (D82)."""
    est = MultiStepValueEstimator(ValueEstimatorConfig(min_samples_for_lcb=2, uncertainty_penalty=0.0))
    for _ in range(3):
        helps = est.open_decision("helps", took_expensive=True)
        est.record_outcome(helps, advantage=0.8)
        hurts = est.open_decision("hurts", took_expensive=True)
        est.record_outcome(hurts, advantage=-0.8)
    assert est.value_lcb("helps") > 0
    assert est.value_lcb("hurts") < 0


def test_unknown_bucket_is_none_not_zero():
    est = MultiStepValueEstimator(ValueEstimatorConfig(min_samples_for_lcb=3))
    assert est.value_lcb("never_seen") is None


def test_stale_pending_is_discarded():
    est = MultiStepValueEstimator(ValueEstimatorConfig(pending_ttl_seconds=0.0))
    decision_id = est.open_decision("b", took_expensive=True)
    import time
    time.sleep(0.01)
    dropped = est.discard_stale()
    assert dropped == 1
    # reporting after discard is a safe no-op
    est.record_outcome(decision_id, advantage=1.0)
    assert est.sample_count("b") == 0
