# Phase 3: real local LLM test (qwen2.5:7b via Ollama, shadow-only)

This is the written record of the design the founder specified, so the
report afterward can be checked against what was actually agreed, not
reconstructed from memory.

## Design (as specified)

- **Model**: `qwen2.5:7b` (locally installed, confirmed via `ollama list` on
  `hybrid-lab`), the SAME model for both paths — isolates ArbiterGate's
  routing strategy from any model-capability difference.
- **Quick path**: one model call, short context (spec + candidates only),
  no test evidence shown.
- **Deep path**: wider context — the real result of running every
  candidate against the public tests is shown to the model as evidence;
  if its first choice fails a public test, it gets one real second
  attempt with that failure named, among the remaining candidates.
- **Judge**: hidden tests only, never shown to the model at any point.
  Real subprocess execution, isolated temp dir, 5s timeout per run.
- **Delayed credit**: a decision's real measured advantage is withheld
  from `ArbiterGate` until `--offset` (default 3) further tasks have been
  processed — not reported the instant it's known, by design.
- **Task count**: 60 total, 20 per bucket (`boundary_bug`, `arith_bug`,
  `logic_bug`) — see `PHASE1_REPORT.md`/`PHASE2_REPORT.md` for why this is
  enough to prove integration/safety but not enough for a statistically
  confident "beats baseline" claim; that is NOT the goal of this run.
- **Compute**: CPU only (`num_gpu=0` on every single Ollama call) so this
  never competes with CEL-0 for GPU memory. No API key. No network beyond
  `127.0.0.1:11434`, enforced by `urllib.request.ProxyHandler({})` (never
  routed through any proxy) exactly as `llm_bridge.py` already does.
- **Safety**: the model NEVER generates or runs free code. It only ever
  names a letter among 4 pre-written candidate functions per task. The
  chosen candidate's PRE-WRITTEN source (never anything the model typed)
  is what actually runs, in an isolated temp directory, in a subprocess,
  with a timeout. A malformed/unparseable model reply falls back to a
  fixed, logged default — never executed blindly, never silently retried
  forever.
- **Time cap**: `--max-minutes` (default 100) is a hard wall-clock cap on
  the whole run; on expiry it stops cleanly after the current task and
  still writes a summary over whatever ran. Every task's result is also
  appended to an incremental JSONL log immediately, so nothing is lost if
  the process is interrupted or killed.

## What this run can and cannot show (stated before running, not after)

**Can show:** whether `ArbiterGate` + `ShadowExecutionAdapter` correctly
wire up to a REAL local model, correctly withhold outcomes until the real
delay elapses, correctly isolate per-bucket learning, and do all of this
safely (no code execution beyond vetted candidates) and within a bounded
time/compute budget.

**Cannot show, at n=60:** that the gate statistically beats a hand-written
static rule in this domain — the sample size is explicitly too small for
that claim, consistent with the honesty standard applied in every prior
phase. Any numbers in the resulting report are integration evidence, not
a powered experiment, and will be labeled as such.

## Before running on `hybrid-lab`

1. `python check_ollama_preflight.py` — confirms Ollama is reachable, the
   model is installed, and gives a real measured per-call latency so the
   `--max-minutes` budget can be sanity-checked before committing 60 tasks
   to it.
2. `python -m pytest arbiter_gate/tests/ -q` — confirms the 15 unit/task-bank
   tests still pass on this machine (same code, different host).
3. `python run_phase3_ollama_experiment.py --n-per-bucket 20 --offset 3`
   — the real run. Add `--max-minutes 90` (or whatever fits the founder's
   time budget) explicitly rather than relying on the default.

Everything here calls `127.0.0.1:11434` directly via Python's stdlib
`urllib` — it does NOT touch `AMG_Studio/studio/server.py`, does not open
any Studio UI endpoint, and makes no change to the live chat or to
`v005`/CEL-0 in any way. It is a standalone script next to the existing
`arbiter_gate/` package, matching the "isolated, reviewable, no live
production change" scoping already used for the CEL-0 diagnostic work.
