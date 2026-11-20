# Arbiter Gate

`ArbiterGate` learns, per context bucket, whether an expensive agent path
has *measured* value over a cheap path — and learns it from real,
delayed, multi-step outcomes, not a single-turn heuristic. It never
assumes it can rewind reality to get a true counterfactual: the caller
supplies the measured advantage, however they obtained it (a true
simulator rollback, a paired/shadow execution on a sample of decisions,
or a regression-adjusted estimate).

Zero third-party dependencies. Pure stdlib, in the core library and in
the optional `arbiter_gate.integrations` helpers.

## Install

```bash
pip install arbiter-gate
```

(Not yet published to PyPI — build and install the local wheel for now;
see **Build locally** below.)

## Quickstart

```python
from arbiter_gate import ArbiterGate

gate = ArbiterGate()
decision = gate.decide(bucket="code_review")
result = expensive_fn() if decision.use_expensive else cheap_fn()

# ... later, once the real outcome is known (same turn or many turns later) ...
gate.report_outcome(decision, advantage=measured_expensive_minus_cheap)
```

`advantage` is deliberately the caller's responsibility to compute. The
most common honest way to get it is to actually run both paths on a
sample of decisions and compare:

```python
from arbiter_gate import ArbiterGate, ShadowExecutionAdapter

gate = ArbiterGate()
adapter = ShadowExecutionAdapter(
    gate=gate,
    cheap_fn=run_cheap_path,
    expensive_fn=run_expensive_path,
    cheap_cost=1.0, expensive_cost=2.0,
)
run = adapter.decide_and_run(bucket="code_review")
# run.deployed_payload is what the caller actually gets back;
# run.immediate_advantage (when not None) is ready to report once your
# own delayed-credit window has elapsed -- see gate.report_outcome above.
```

`ShadowExecutionAdapter` only reports a counterfactual when both paths
were actually executed and comparably scored — it will not let a raw
thumbs-up/re-ask signal (which only observes the deployed path) stand in
for a real counterfactual. See `INTEGRATION_GUIDE.md` for the common
pitfalls here and how to avoid them.

## Persisting learned state across runs (v0.3.0+)

By default `ArbiterGate` keeps everything in memory, so a new process
(a new CI run, a new server instance) starts every bucket at zero —
nothing it learns ever carries over, and nothing can ever leak between
runs either, because there's no state to leak. That's a safe default,
but it also means a routing decision in process N+1 never benefits from
what process N already measured.

`arbiter_gate.integrations` adds an explicit, opt-in way to carry that
evidence across runs via a plain JSON file:

```python
from arbiter_gate.integrations import save_state_atomic, load_state_or_new

gate, report = load_state_or_new("gate_state.json", repo_id="org/repo")
# report == {"loaded": True/False, "reason": "..."} -- log it, don't guess
... use gate.decide / gate.report_outcome as normal ...
save_state_atomic(gate, "gate_state.json", repo_id="org/repo")
```

- **What persists**: the exploration counter per bucket, the settled
  value-estimator statistics (mean/count/variance), and the cooldown
  state. **What never persists**: open decision tickets that haven't
  been reported or abandoned yet — those are tied to the current
  process's clock and TTL window, so each run closes its own tickets;
  only settled evidence crosses the boundary.
- **Crash-safety**: `save_state_atomic` writes to a temp file and
  `os.replace()`s it over the destination, which is atomic on POSIX and
  Windows. An interruption mid-write leaves the previous file (or no
  file, on a first save) fully intact — never a half-written one.
- **Per-repo isolation**: the saved file is stamped with the `repo_id`
  you pass in. Loading it with a different `repo_id` refuses to apply
  it (falls back to a fresh gate by default; `load_state_strict` raises
  `StateIntegrityError` instead), so a copied or mis-pathed file can't
  silently contaminate a different repo's learned routing.
- **What this does *not* claim**: that learning actually improves
  real-world cost or quality over time at scale — only that evidence
  correctly survives a process boundary. See
  `arbiter-gate-demo-target`'s own evidence ladder for where that larger
  claim currently stands (as of this release: mechanism verified,
  continuing-benefit-at-scale not yet measured).

## What's actually validated, and what isn't yet

This isn't a claim made from first principles — it's the result of six
rounds of validation (Phase 0 through Phase 6), each pre-registered
before looking at results, each independently re-verified from raw data
rather than taken on the generator's word. Phase reports are in this
repo (`PHASE1_REPORT.md` ... `PHASE6_REPORT.md`); the honest summary:

- **Mechanics** (routing, delayed credit, cooldown-on-harm, shadow
  execution): validated on two deterministic domains (Phase 0–2) and
  against a real local LLM (`qwen2.5:7b` via Ollama) on both CPU (Phase 3)
  and GPU (Phase 4–6).
- **Performance claim** (does the gate actually beat a fixed policy):
  Phase 4 came close but narrowly missed its own pre-registered bar.
  Phase 5 appeared to pass but the result was confounded by its own
  experiment configuration (disclosed and corrected, not hidden —
  see `PHASE5_REPORT.md`). **Phase 6, after fixing that confound, is the
  first run that genuinely clears the bar**: +0.080 quality over an
  always-quick policy and 16.7% lower latency than an always-expensive
  policy, on 600 held-out tasks, with the gain confirmed to hold even
  when forced-exploration decisions are excluded from the comparison.
- **Not yet tested**: a genuinely different model for the cheap vs.
  expensive path (Phase 3–6 used the same model for both, specifically to
  isolate the gating strategy from model-capability differences), and
  task families beyond code-repair candidate selection.

Every reserved validation seed used so far (10–14, 15–19, 20–24) is
consumed and must not be reused for a new performance claim; a further
test needs fresh seeds and a fresh pre-registration.

## Build locally

```bash
python -m pip install build
python -m build          # produces dist/*.whl and dist/*.tar.gz
pip install dist/arbiter_gate-*.whl
```

## Verify

```bash
python -m pytest tests -v
python -m arbiter_gate.bench.code_repair_pipeline 10 11 12 13 14
python -m arbiter_gate.bench.multihop_qa_agent 10 11 12 13 14
```

The first benchmark should structurally match `phase1_validation_results.json`;
the second should match `phase2_validation_results.json`. `bench/` holds
this project's own research/validation harnesses (including the Phase 3–6
real-LLM experiment drivers) — useful for reproducing the published
results, but it ships with the repo, not with the installed package, so
`pip install arbiter-gate` stays minimal and dependency-free.

## Executable DevOps proof

`examples/devops_live_proof/` turns the routing claim into CI evidence.
It runs cheap and deep code candidates in fresh temporary projects, grades
them with real pytest processes, reports only measured counterfactuals, and
uploads `summary.json` plus the full `decision_audit.jsonl` from GitHub
Actions. The fixtures are deterministic and do not pretend to be LLM output;
their purpose is to make the gate's routing, measurement, and audit behavior
independently reproducible without credentials or inference cost.

```bash
python examples/devops_live_proof/run_proof.py --output live-proof-artifacts
```

See `examples/devops_live_proof/README.md` for the exact cases and expected
learned routes.

## Real-agent integration

See `INTEGRATION_GUIDE.md`. Provider model identifiers and credentials
belong to the caller. This package does not read, persist, or transmit
credentials.

## Safety

`arbiter_gate.integrations.ollama_client` only ever returns the raw text
a model produced — it never executes anything. In every validation phase
so far, the caller parses that text as a single letter choice among
pre-written, vetted candidates and runs the chosen candidate in an
isolated subprocess with a timeout; free-form model-generated code is
never executed. This is a property of how this repo's own benchmarks use
the library, not an enforced constraint of the library itself — if you
build a different integration, that safety boundary is yours to keep.

## License

Apache License 2.0 — see [LICENSE](LICENSE). This covers the `arbiter_gate` library (core/adapters/integrations). The "Arbiter Gate" name and brand are kept separate from the code license: Apache-2.0 already excludes trademark rights by its own terms (Section 6), so no additional wrapper is needed to keep a future commercial/hosted offering distinct from the open library.
