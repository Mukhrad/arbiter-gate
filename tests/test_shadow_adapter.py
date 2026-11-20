"""Tests for the Phase 1 shadow-execution adapter: the core honesty
constraint is that an outcome is only ever reported when both paths were
actually executed (exploring/probe), and never fabricated for trusted or
blocked decisions."""
from arbiter_gate.adapters.shadow import ShadowExecutionAdapter
from arbiter_gate.core.gate import ArbiterGate, ArbiterGateConfig
from arbiter_gate.core.value_estimator import ValueEstimatorConfig


def _counting_fns():
    calls = {"cheap": 0, "expensive": 0}

    def cheap():
        calls["cheap"] += 1
        return "cheap_payload", 0.3

    def expensive():
        calls["expensive"] += 1
        return "expensive_payload", 0.9

    return cheap, expensive, calls


def test_exploring_decision_runs_both_paths_for_real():
    gate = ArbiterGate(ArbiterGateConfig(exploratory_decisions=4))
    cheap, expensive, calls = _counting_fns()
    adapter = ShadowExecutionAdapter(gate, cheap, expensive)

    run = adapter.decide_and_run("bucket_a")

    assert run.decision.reason == "exploring"
    assert calls["cheap"] == 1 and calls["expensive"] == 1
    assert run.immediate_advantage is not None
    assert abs(run.immediate_advantage - (0.9 - 0.3)) < 1e-9


def test_trusted_decision_runs_only_the_chosen_path():
    # min_samples_for_lcb default is 3, so feed enough consistently positive
    # exploring outcomes to actually cross into "trusted" before checking
    # the single-execution invariant there.
    gate = ArbiterGate(ArbiterGateConfig(exploratory_decisions=4))
    cheap, expensive, calls = _counting_fns()
    adapter = ShadowExecutionAdapter(gate, cheap, expensive)

    for _ in range(4):
        run = adapter.decide_and_run("bucket_b")
        adapter.report_delayed(run.decision, run.immediate_advantage)

    calls["cheap"] = 0
    calls["expensive"] = 0
    run = adapter.decide_and_run("bucket_b")

    assert run.decision.reason == "trusted"
    # Only the deployed path should have actually executed.
    assert calls["expensive"] == 1
    assert calls["cheap"] == 0
    assert run.shadow_payload is None
    assert run.immediate_advantage is None


def test_no_outcome_is_fabricated_when_only_one_path_ran():
    gate = ArbiterGate(ArbiterGateConfig(exploratory_decisions=0))
    cheap, expensive, calls = _counting_fns()
    adapter = ShadowExecutionAdapter(gate, cheap, expensive)

    # With 0 exploratory decisions, the very first decision has no LCB yet
    # (None), so it should be "blocked" (cheap-only), not "trusted".
    run = adapter.decide_and_run("bucket_c")
    assert run.decision.reason == "blocked"
    assert run.immediate_advantage is None
    assert run.shadow_payload is None
    assert calls["expensive"] == 0
    assert calls["cheap"] == 1
    assert gate.diagnostics()["value"]["pending_open"] == 0


def test_trusted_single_path_closes_unmeasurable_ticket():
    gate = ArbiterGate(
        ArbiterGateConfig(
            exploratory_decisions=2,
            value_config=ValueEstimatorConfig(min_samples_for_lcb=2),
        )
    )
    cheap, expensive, _ = _counting_fns()
    adapter = ShadowExecutionAdapter(gate, cheap, expensive)

    for _ in range(2):
        run = adapter.decide_and_run("trusted_bucket")
        adapter.report_delayed(run.decision, run.immediate_advantage)

    run = adapter.decide_and_run("trusted_bucket")
    assert run.decision.reason == "trusted"
    assert gate.diagnostics()["value"]["pending_open"] == 0
