"""Cooldown + sparse-probe authority control, per context bucket.

Same spirit as BioMind-X4's CausalValuePlanner (negative streak -> cooldown,
periodic probe to avoid permanent lockout), generalized to be per-bucket and
decoupled from any single-step assumption: this module only reacts to
LCB/mean values handed to it by the value estimator, whatever horizon those
were computed over.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CooldownConfig:
    negative_streak_for_cooldown: int = 2
    cooldown_decisions: int = 8       # measured in decisions on this bucket, not wall time
    probe_interval: int = 10          # 1 forced probe every N consecutive blocks


@dataclass
class _BucketCooldownState:
    negative_streak: int = 0
    cooldown_remaining: int = 0
    blocked_in_a_row: int = 0


class CooldownController:
    def __init__(self, config: CooldownConfig | None = None) -> None:
        self.config = config or CooldownConfig()
        self._state: dict[str, _BucketCooldownState] = {}

    def _get(self, bucket: str) -> _BucketCooldownState:
        return self._state.setdefault(bucket, _BucketCooldownState())

    def note_realized_value(self, bucket: str, advantage: float, margin: float) -> None:
        """Call once per realized outcome (same timing as
        ValueEstimator.record_outcome) to update the negative streak."""
        state = self._get(bucket)
        if advantage < -margin:
            state.negative_streak += 1
        else:
            state.negative_streak = 0
        if state.negative_streak >= self.config.negative_streak_for_cooldown:
            state.cooldown_remaining = self.config.cooldown_decisions
            state.negative_streak = 0

    def allow_despite_block(self, bucket: str) -> bool:
        """Called only when the value estimate says 'block'. Returns True on
        a sparse forced probe, so a bucket that was blocked for a stale
        reason (context shifted) can still recover."""
        state = self._get(bucket)
        if state.cooldown_remaining > 0:
            state.cooldown_remaining -= 1
            return False
        state.blocked_in_a_row += 1
        if state.blocked_in_a_row % max(1, self.config.probe_interval) == 0:
            return True
        return False

    def note_allowed(self, bucket: str) -> None:
        self._get(bucket).blocked_in_a_row = 0

    def diagnostics(self) -> dict:
        return {
            b: {"cooldown_remaining": s.cooldown_remaining,
                "negative_streak": s.negative_streak,
                "blocked_in_a_row": s.blocked_in_a_row}
            for b, s in self._state.items()
        }
