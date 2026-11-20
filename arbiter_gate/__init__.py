"""Arbiter Gate: measured counterfactual-advantage gating between a cheap
and an expensive agent path, with delayed/multi-step outcome reporting and
automatic cooldown on harm.

Quickstart:

    from arbiter_gate import ArbiterGate

    gate = ArbiterGate()
    decision = gate.decide(bucket="code_review")
    result = expensive_fn() if decision.use_expensive else cheap_fn()
    ... later, once the real outcome is known ...
    gate.report_outcome(decision, advantage=measured_expensive_minus_cheap)

For the common case where "the real outcome" requires actually running
both paths to measure (paired/shadow execution), see
`arbiter_gate.adapters.ShadowExecutionAdapter`.

Zero third-party dependencies (stdlib only) in the core library and the
optional `arbiter_gate.integrations` helpers.
"""
from .core import ArbiterGate, ArbiterGateConfig, GateDecision
from .adapters import ShadowExecutionAdapter, ShadowRunResult

__version__ = "0.3.0"

__all__ = [
    "ArbiterGate", "ArbiterGateConfig", "GateDecision",
    "ShadowExecutionAdapter", "ShadowRunResult",
    "__version__",
]
