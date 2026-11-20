"""Measured counterfactual value estimation — multi-step/delayed by design.

This directly fixes the disclosed flaw in BioMind-X4 (AMG project, P-20261118-41,
decision D82): that gate measured the value of a planner's intervention by
comparing it to a reflex counterfactual over exactly ONE environment step. On
four of five seeds the measured value sat at noise level regardless of whether
the intervention actually helped or hurt over the full episode, because a
one-step comparison cannot see value that only materializes later.

The fix here is a ticket/callback pattern, not a bigger fixed horizon: the gate
hands out an open `decision_id` at decision time, and the caller reports the
realized outcome whenever it is actually known -- one step later, k steps
later, or at the end of an episode/conversation/task. This works whether the
caller can cheaply simulate a true counterfactual (rare, mostly in simulators)
or can only observe a paired/shadow run's outcome after a real delay (the
common case for real AI-agent deployments, where you cannot rewind reality).

Design notes:
- Value is tracked per *context bucket* (a hashable key the caller supplies,
  e.g. a coarse embedding cluster id or a task-type label), not globally --
  a mechanism that helps on one kind of query and hurts on another must not
  average out to "roughly zero" the way X4's global-ish measurement did.
- Uncertainty is tracked (Welford's online variance) so authority is granted
  on a lower confidence bound, not a noisy point estimate -- same spirit as
  X4's `value_lcb`, generalized to per-bucket.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
import uuid


@dataclass
class _BucketStats:
    mean: float = 0.0
    count: int = 0
    _m2: float = 0.0  # sum of squared deviations, for Welford variance

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self._m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self._m2 / (self.count - 1)

    @property
    def stderr(self) -> float:
        if self.count == 0:
            return float("inf")
        return math.sqrt(max(0.0, self.variance) / self.count)


@dataclass
class PendingDecision:
    decision_id: str
    bucket: str
    took_expensive: bool
    created_at: float
    expired: bool = False


@dataclass
class ValueEstimatorConfig:
    uncertainty_penalty: float = 1.0  # multiplies stderr when forming the LCB
    min_samples_for_lcb: int = 3       # below this, treat as "unknown", not "bad"
    value_margin: float = 0.0          # LCB must exceed this to grant authority
    pending_ttl_seconds: float = 3600.0  # outcomes reported after this are dropped as stale


class MultiStepValueEstimator:
    """Tracks measured advantage of 'expensive' over 'cheap', per context
    bucket, fed by outcomes that may arrive long after the decision was made.
    """

    def __init__(self, config: ValueEstimatorConfig | None = None) -> None:
        self.config = config or ValueEstimatorConfig()
        self._stats: dict[str, _BucketStats] = {}
        self._pending: dict[str, PendingDecision] = {}

    def open_decision(self, bucket: str, took_expensive: bool) -> str:
        """Call when a routing decision is made. Returns a decision_id to pass
        to `record_outcome` once the realized value is known (immediately, or
        arbitrarily later)."""
        decision_id = uuid.uuid4().hex
        self._pending[decision_id] = PendingDecision(
            decision_id=decision_id, bucket=bucket,
            took_expensive=took_expensive, created_at=time.monotonic(),
        )
        return decision_id

    def record_outcome(self, decision_id: str, advantage: float) -> None:
        """`advantage` is the realized (expensive - cheap) outcome difference
        for this one decision, already computed by the caller however it can:
        a true counterfactual rollout, a paired shadow execution, or a
        regression-adjusted estimate. Only used if the decision actually took
        the expensive path and is still open (not expired, not double-reported)."""
        pending = self._pending.pop(decision_id, None)
        if pending is None or pending.expired:
            return
        if time.monotonic() - pending.created_at > self.config.pending_ttl_seconds:
            return
        if not pending.took_expensive:
            # Advantage is only meaningful relative to a baseline we didn't
            # take; recording it for a cheap-path decision would conflate
            # "we never tested expensive here" with "expensive measured bad".
            return
        self._stats.setdefault(pending.bucket, _BucketStats()).update(advantage)

    def abandon_decision(self, decision_id: str) -> bool:
        """Close a ticket when no counterfactual will ever be measured.

        Production adapters use this after a single-path decision. Keeping
        such tickets open until their TTL would not add evidence; it would
        only inflate diagnostics and retain avoidable state.
        """
        return self._pending.pop(decision_id, None) is not None

    def discard_stale(self) -> int:
        """Drop pending decisions past their TTL (caller never reported an
        outcome). Returns how many were dropped. Call periodically."""
        now = time.monotonic()
        stale = [k for k, p in self._pending.items()
                 if now - p.created_at > self.config.pending_ttl_seconds]
        for k in stale:
            del self._pending[k]
        return len(stale)

    def value_lcb(self, bucket: str) -> float | None:
        """Lower confidence bound on measured advantage for this bucket.
        Returns None if there are not yet enough samples to say anything --
        callers should treat None as 'unknown', distinct from 'measured and
        non-positive'."""
        stats = self._stats.get(bucket)
        if stats is None or stats.count < self.config.min_samples_for_lcb:
            return None
        return stats.mean - self.config.uncertainty_penalty * stats.stderr

    def sample_count(self, bucket: str) -> int:
        stats = self._stats.get(bucket)
        return stats.count if stats else 0

    def mean(self, bucket: str) -> float:
        stats = self._stats.get(bucket)
        return stats.mean if stats else 0.0

    def diagnostics(self) -> dict:
        return {
            "buckets": {
                b: {"mean": s.mean, "count": s.count, "stderr": s.stderr,
                    "lcb": self.value_lcb(b)}
                for b, s in self._stats.items()
            },
            "pending_open": len(self._pending),
        }
