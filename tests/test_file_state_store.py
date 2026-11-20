from __future__ import annotations

import os

import pytest

from arbiter_gate.core import ArbiterGate, ArbiterGateConfig, ValueEstimatorConfig
from arbiter_gate.integrations import (
    save_state_atomic, load_state_or_new, load_state_strict, StateIntegrityError,
)


def _confident_config() -> ArbiterGateConfig:
    return ArbiterGateConfig(
        exploratory_decisions=2,
        value_config=ValueEstimatorConfig(min_samples_for_lcb=2, uncertainty_penalty=0.5),
    )


def _trusted_gate() -> ArbiterGate:
    gate = ArbiterGate(_confident_config())
    for d in [gate.decide("b") for _ in range(2)]:
        gate.report_outcome(d, advantage=0.5)
    assert gate.decide("b").reason == "trusted"
    return gate


def test_save_then_load_restores_trust_through_real_files(tmp_path):
    path = tmp_path / "state.json"
    save_state_atomic(_trusted_gate(), path, repo_id="org/repo")

    loaded, report = load_state_or_new(path, repo_id="org/repo", config=_confident_config())
    assert report == {"loaded": True, "reason": "ok"}
    assert loaded.decide("b").reason == "trusted"


def test_missing_file_returns_fresh_gate_not_an_error(tmp_path):
    gate, report = load_state_or_new(tmp_path / "never_written.json", repo_id="org/repo")
    assert report["loaded"] is False
    assert report["reason"] == "no_state_file"
    assert gate.decide("b").reason == "exploring"  # genuinely fresh, no crash


def test_repo_id_mismatch_is_isolated_not_silently_applied(tmp_path):
    path = tmp_path / "state.json"
    save_state_atomic(_trusted_gate(), path, repo_id="org/repo-a")

    # Same file, wrong repo_id: must NOT silently inherit repo-a's trust
    gate, report = load_state_or_new(path, repo_id="org/repo-b", config=_confident_config())
    assert report["loaded"] is False
    assert "repo_id_mismatch" in report["reason"]
    assert gate.decide("b").reason == "exploring"  # falls back to a genuinely fresh gate

    with pytest.raises(StateIntegrityError):
        load_state_strict(path, repo_id="org/repo-b")

    # the correct repo_id still works on the very same file
    same_repo_gate = load_state_strict(path, repo_id="org/repo-a", config=_confident_config())
    assert same_repo_gate.decide("b").reason == "trusted"


def test_corrupt_file_falls_back_safely_without_crashing(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not valid json at all", encoding="utf-8")

    gate, report = load_state_or_new(path, repo_id="org/repo")
    assert report["loaded"] is False
    assert "unreadable_state_file" in report["reason"]
    assert gate.decide("b").reason == "exploring"


def test_atomic_save_leaves_no_temp_files_behind(tmp_path):
    path = tmp_path / "state.json"
    save_state_atomic(_trusted_gate(), path, repo_id="org/repo")
    leftovers = [p for p in os.listdir(tmp_path) if p != path.name]
    assert leftovers == []


def test_interrupted_write_does_not_corrupt_the_previous_state(tmp_path, monkeypatch):
    """Simulates a crash between writing the temp file and the atomic
    rename: the previously-saved state must remain fully intact and loadable
    -- this is the 'recovery after interruption' guarantee."""
    path = tmp_path / "state.json"
    save_state_atomic(_trusted_gate(), path, repo_id="org/repo")
    original_bytes = path.read_bytes()

    def boom(*args, **kwargs):
        raise OSError("simulated crash before the atomic replace completes")

    monkeypatch.setattr(os, "replace", boom)
    fresh_untrusted = ArbiterGate(_confident_config())  # a second, different gate to try saving
    with pytest.raises(OSError):
        save_state_atomic(fresh_untrusted, path, repo_id="org/repo")
    monkeypatch.undo()

    # the destination file must be byte-for-byte what it was before the
    # failed save, and no stray temp file should remain
    assert path.read_bytes() == original_bytes
    leftovers = [p for p in os.listdir(tmp_path) if p != path.name]
    assert leftovers == []

    gate, report = load_state_or_new(path, repo_id="org/repo", config=_confident_config())
    assert report["loaded"] is True
    assert gate.decide("b").reason == "trusted"  # recovered exactly the pre-crash state
