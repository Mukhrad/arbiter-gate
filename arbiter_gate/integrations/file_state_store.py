"""File-backed persistence for ArbiterGate state, with atomic writes and
per-repo isolation.

This exists to answer a concrete, previously-open gap: without it, every
process that constructs an ArbiterGate (e.g. a fresh GitHub Actions runner on
every CI run) starts from zero -- it must re-explore every bucket from
scratch, and no routing decision ever benefits from a prior run's evidence.
That is safe (nothing can leak) but makes it impossible to measure any
real, continuing benefit from the gate's learning.

This module does not change that default: ArbiterGate itself still has no
built-in persistence, by design (callers who don't want file state keep the
exact behavior they had before). This is an explicit opt-in a caller wires
up itself, choosing its own file path -- which is also how per-repo isolation
is enforced: there is one state file per repo, and this module additionally
stamps and checks a `repo_id` inside the file itself, so a state file
produced for one repo can never be silently applied to a different one, even
if a path collision or a careless copy puts the wrong file in the wrong
place.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..core.gate import ArbiterGate, ArbiterGateConfig


class StateIntegrityError(Exception):
    """Raised by `load_state_strict` when a state file's recorded repo_id
    does not match the repo_id the caller expects."""


def save_state_atomic(gate: ArbiterGate, path: str | os.PathLike, repo_id: str) -> None:
    """Write `gate`'s learned state to `path`, tagged with `repo_id`.

    Atomic: writes to a sibling temp file in the same directory, fsyncs it,
    then os.replace()s it over the destination. os.replace is atomic on both
    POSIX and Windows, so a crash or interruption mid-write leaves the
    previous file (or no file, on a first save) intact -- the destination
    path is never observed half-written. `load_state_or_new` additionally
    treats a state file that is corrupt for any other reason (truncated by
    something outside this module, hand-edited, etc.) as "no usable state"
    rather than raising, so a process restarting after an interruption
    always has a safe path forward: either the last fully-written state, or
    a fresh gate -- never a crash on load.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"repo_id": repo_id, "gate_state": gate.export_state()}
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def load_state_or_new(
    path: str | os.PathLike,
    repo_id: str,
    config: ArbiterGateConfig | None = None,
) -> tuple[ArbiterGate, dict]:
    """Load `path`'s saved state into a fresh ArbiterGate, if the file is
    present, readable, schema-valid, and tagged with this same `repo_id`.
    Otherwise return a fresh, empty gate.

    Returns (gate, report), where report is
    {"loaded": bool, "reason": str} so a caller can log plainly what
    actually happened on this run instead of silently assuming persistence
    worked.

    Deliberately never raises for a missing, corrupt, or foreign-repo file:
    "no usable prior state" is always a safe (if less-optimized) degraded
    mode for a routing gate -- it falls back to exploring, not to a crash.
    Use `load_state_strict` instead if a repo_id mismatch specifically should
    halt the caller rather than silently fall back.
    """
    path = Path(path)
    if not path.exists():
        return ArbiterGate(config), {"loaded": False, "reason": "no_state_file"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        return ArbiterGate(config), {"loaded": False, "reason": f"unreadable_state_file: {e}"}

    # json.load happily returns a list, string, number, etc. for valid JSON
    # that isn't an object -- e.g. a file containing just "[]". Reject that
    # explicitly rather than calling .get() on it and crashing with an
    # unhandled AttributeError (found and fixed after an independent
    # reviewer reproduced exactly this with an empty-list state file).
    if not isinstance(payload, dict):
        return ArbiterGate(config), {
            "loaded": False,
            "reason": f"malformed_state_file: expected a JSON object at the top level, got {type(payload).__name__}",
        }

    stored_repo_id = payload.get("repo_id")
    if stored_repo_id != repo_id:
        return ArbiterGate(config), {
            "loaded": False,
            "reason": f"repo_id_mismatch: file is for {stored_repo_id!r}, expected {repo_id!r}",
        }

    gate_state = payload.get("gate_state")
    if not isinstance(gate_state, dict):
        return ArbiterGate(config), {
            "loaded": False,
            "reason": f"malformed_state_file: 'gate_state' is not a JSON object (got {type(gate_state).__name__})",
        }

    # ArbiterGate.load_state is itself all-or-nothing (see its docstring), so
    # `gate` below is guaranteed untouched if it raises. We still return a
    # brand-new ArbiterGate on failure rather than the attempted one, as a
    # second, independent line of defense against ever returning a
    # partially-mutated gate from this function, no matter how load_state's
    # own internals might change in the future.
    gate = ArbiterGate(config)
    try:
        gate.load_state(gate_state)
    except (KeyError, ValueError, TypeError) as e:
        return ArbiterGate(config), {"loaded": False, "reason": f"invalid_gate_state: {e}"}
    return gate, {"loaded": True, "reason": "ok"}


def load_state_strict(
    path: str | os.PathLike,
    repo_id: str,
    config: ArbiterGateConfig | None = None,
) -> ArbiterGate:
    """Like `load_state_or_new`, but raises StateIntegrityError on a
    repo_id mismatch instead of silently falling back to a fresh gate -- for
    callers that would rather halt than route on a possibly
    cross-repo-contaminated assumption. A missing or corrupt file still
    returns a fresh gate (that is absence of history, not contamination)."""
    gate, report = load_state_or_new(path, repo_id, config)
    if not report["loaded"] and report["reason"].startswith("repo_id_mismatch"):
        raise StateIntegrityError(report["reason"])
    return gate
