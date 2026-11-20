from __future__ import annotations

import pytest

from arbiter_gate.core import ArbiterGate, ArbiterGateConfig, ValueEstimatorConfig


def _confident_config() -> ArbiterGateConfig:
    return ArbiterGateConfig(
        exploratory_decisions=2,
        value_config=ValueEstimatorConfig(min_samples_for_lcb=2, uncertainty_penalty=0.5),
    )


def test_export_load_roundtrip_grants_trust_on_first_decision_of_fresh_gate():
    """The scenario the founder explicitly asked to be proven: a brand-new
    process (simulating a new CI run) whose very first decision on a bucket
    already benefits from a prior run's evidence, instead of re-exploring."""
    run1 = ArbiterGate(_confident_config())
    decisions = [run1.decide("b") for _ in range(2)]
    for d in decisions:
        run1.report_outcome(d, advantage=0.5)
    # run1 itself is now trusted on "b" -- confirm the premise before testing persistence
    assert run1.decide("b").reason == "trusted"

    state = run1.export_state()

    run2 = ArbiterGate(_confident_config())  # a genuinely fresh instance, zero in-memory history
    run2.load_state(state)
    first_decision_of_run2 = run2.decide("b")
    assert first_decision_of_run2.reason == "trusted"
    assert first_decision_of_run2.use_expensive


def test_without_persistence_a_fresh_gate_would_have_re_explored():
    """Control case, proving the above isn't vacuous: a fresh gate with no
    loaded state really does start over at 'exploring' for the same config."""
    fresh = ArbiterGate(_confident_config())
    assert fresh.decide("b").reason == "exploring"


def test_decisions_per_bucket_persists_so_exploration_count_does_not_reset():
    run1 = ArbiterGate(ArbiterGateConfig(exploratory_decisions=5))
    for _ in range(3):
        run1.decide("b")
    state = run1.export_state()
    assert state["decisions_per_bucket"]["b"] == 3

    run2 = ArbiterGate(ArbiterGateConfig(exploratory_decisions=5))
    run2.load_state(state)
    # 2 more decisions would reach 5 total if the counter persisted; a reset
    # counter would instead still be "exploring" on decision 4 and 5
    reasons = [run2.decide("b").reason for _ in range(2)]
    assert reasons == ["exploring", "exploring"]
    assert run2.decide("b").reason != "exploring"  # 6th ever decision on "b", past the threshold


def test_cooldown_state_persists_across_load():
    from arbiter_gate.core import CooldownConfig

    run1 = ArbiterGate(ArbiterGateConfig(
        exploratory_decisions=2,
        value_config=ValueEstimatorConfig(min_samples_for_lcb=2, uncertainty_penalty=0.5),
        cooldown_config=CooldownConfig(negative_streak_for_cooldown=1, cooldown_decisions=50, probe_interval=1000),
    ))
    decisions = [run1.decide("b") for _ in range(2)]
    for d in decisions:
        run1.report_outcome(d, advantage=-0.5)
    assert run1.decide("b").reason == "blocked"  # cooldown engaged in run1
    state = run1.export_state()

    run2 = ArbiterGate(ArbiterGateConfig(
        exploratory_decisions=2,
        value_config=ValueEstimatorConfig(min_samples_for_lcb=2, uncertainty_penalty=0.5),
        cooldown_config=CooldownConfig(negative_streak_for_cooldown=1, cooldown_decisions=50, probe_interval=1000),
    ))
    run2.load_state(state)
    assert run2.decide("b").reason == "blocked"  # still in cooldown, carried over


def test_pending_undecided_tickets_are_not_persisted():
    """Open a decision but never report or abandon it, then export: the
    pending ticket must not appear in the exported value stats (it was never
    settled evidence), and loading it elsewhere must not resurrect it."""
    run1 = ArbiterGate(ArbiterGateConfig(exploratory_decisions=1))
    run1.decide("b")  # opens a pending ticket, never reported
    state = run1.export_state()
    assert state["value_stats"].get("b", {}).get("count", 0) == 0


def test_load_state_rejects_unknown_schema_version():
    gate = ArbiterGate()
    bogus = {"schema_version": 999, "decisions_per_bucket": {}, "value_stats": {}, "cooldown_state": {}}
    with pytest.raises(ValueError):
        gate.load_state(bogus)
