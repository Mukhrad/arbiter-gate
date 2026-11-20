"""Regression tests for two bugs an independent reviewer (CN-003) found and
reproduced in the first persistence implementation:

1. A state file that is valid JSON but not an object at the top level (the
   reviewer's example: a file containing just `[]`) crashed with an
   unhandled AttributeError instead of being treated as "no usable state".
2. ArbiterGate.load_state applied decisions_per_bucket, then value_stats,
   then cooldown_state one at a time. If value_stats or cooldown_state
   failed to validate, decisions_per_bucket had already been mutated --
   leaving the gate with an advanced exploration counter but no real
   evidence behind it, so the next decision on that bucket would skip
   exploration with nothing to justify it.

Both are fixed by: rejecting a non-dict top-level payload explicitly, and
by validating every field of a load_state call before mutating anything on
the gate (true all-or-nothing).
"""
from __future__ import annotations

import json
import math

import pytest

from arbiter_gate.core import ArbiterGate, ArbiterGateConfig
from arbiter_gate.integrations import load_state_or_new


def test_top_level_json_list_is_rejected_not_crashed(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("[]", encoding="utf-8")

    gate, report = load_state_or_new(path, repo_id="org/repo")
    assert report["loaded"] is False
    assert "malformed_state_file" in report["reason"]
    assert gate.decide("b").reason == "exploring"  # genuinely fresh, no crash


def test_gate_state_that_is_not_an_object_is_rejected_not_crashed(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"repo_id": "org/repo", "gate_state": "not an object"}), encoding="utf-8")

    gate, report = load_state_or_new(path, repo_id="org/repo")
    assert report["loaded"] is False
    assert "malformed_state_file" in report["reason"]
    assert gate.decide("b").reason == "exploring"


def test_advanced_counter_with_empty_value_stats_does_not_skip_exploration(tmp_path):
    """The reviewer's exact second repro case: decisions_per_bucket says a
    bucket was decided 100 times, but its value_stats entry is `{}` (missing
    required fields). This must be rejected as a whole, not partially
    applied -- the next decision on that bucket must still explore, not
    jump straight to a stale 'blocked' with no real evidence."""
    path = tmp_path / "state.json"
    payload = {
        "repo_id": "org/repo",
        "gate_state": {
            "schema_version": 1,
            "decisions_per_bucket": {"b": 100},
            "value_stats": {"b": {}},  # missing mean/count/m2 -> KeyError
            "cooldown_state": {},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    gate, report = load_state_or_new(path, repo_id="org/repo", config=ArbiterGateConfig(exploratory_decisions=2))
    assert report["loaded"] is False
    assert "invalid_gate_state" in report["reason"]
    assert gate.decisions_per_bucket == {}  # NOT left at the stale 100
    assert gate.decide("b").reason == "exploring"  # NOT jumped to "blocked" with no evidence


def test_malformed_value_stats_do_not_leak_a_stale_decision_counter_via_direct_api():
    """Same corruption, called directly on ArbiterGate.load_state (not
    through the file store), to confirm the gate itself -- not just the
    file-store wrapper -- never partially applies a load."""
    gate = ArbiterGate(ArbiterGateConfig(exploratory_decisions=2))
    bad_state = {
        "schema_version": 1,
        "decisions_per_bucket": {"b": 100},
        "value_stats": {"b": {}},
        "cooldown_state": {},
    }
    with pytest.raises(KeyError):
        gate.load_state(bad_state)
    assert gate.decisions_per_bucket == {}
    assert gate.decide("b").reason == "exploring"


def test_negative_count_is_rejected_not_silently_accepted():
    gate = ArbiterGate()
    bad_state = {
        "schema_version": 1,
        "decisions_per_bucket": {},
        "value_stats": {"b": {"mean": 5.0, "count": -3, "m2": 0.0}},
        "cooldown_state": {},
    }
    with pytest.raises(ValueError):
        gate.load_state(bad_state)
    assert gate.value.sample_count("b") == 0


def test_nan_mean_is_rejected_not_silently_accepted_as_trusted():
    """A NaN mean must never be loaded at all -- not merely 'happen not to
    compare as > margin'. Rejecting the whole state outright is the only
    guarantee that corrupted statistics can never manufacture a 'trusted'
    decision, regardless of how value_margin or the LCB comparison logic
    might change in the future."""
    gate = ArbiterGate()
    bad_state = {
        "schema_version": 1,
        "decisions_per_bucket": {"b": 10},
        "value_stats": {"b": {"mean": float("nan"), "count": 50, "m2": 0.0}},
        "cooldown_state": {},
    }
    with pytest.raises(ValueError):
        gate.load_state(bad_state)
    assert gate.value.sample_count("b") == 0
    assert gate.decisions_per_bucket == {}


def test_negative_cooldown_fields_are_rejected():
    gate = ArbiterGate()
    bad_state = {
        "schema_version": 1,
        "decisions_per_bucket": {},
        "value_stats": {},
        "cooldown_state": {"b": {"negative_streak": -1, "cooldown_remaining": 0, "blocked_in_a_row": 0}},
    }
    with pytest.raises(ValueError):
        gate.load_state(bad_state)


def test_cross_component_atomicity_bad_cooldown_does_not_leave_value_stats_applied():
    """If decisions_per_bucket and value_stats are valid but cooldown_state
    is not, NONE of the three should be applied -- not just cooldown."""
    gate = ArbiterGate(ArbiterGateConfig(exploratory_decisions=2))
    bad_state = {
        "schema_version": 1,
        "decisions_per_bucket": {"b": 5},
        "value_stats": {"b": {"mean": 1.0, "count": 5, "m2": 0.0}},
        "cooldown_state": {"b": {"negative_streak": -1, "cooldown_remaining": 0, "blocked_in_a_row": 0}},
    }
    with pytest.raises(ValueError):
        gate.load_state(bad_state)
    assert gate.decisions_per_bucket == {}
    assert gate.value.sample_count("b") == 0  # value_stats must not have been applied either
