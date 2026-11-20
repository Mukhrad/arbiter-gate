# Phase 1 validation: real shadow execution, real delayed outcomes

## Why this phase exists

`VALIDATION_REPORT.md` (Phase 0) was explicit about its own limit:

> "advantage is handed by an omniscient benchmark harness, not recovered
> from real paired/shadow execution — that's Phase 1's job."

Phase 0's synthetic benchmark fed the gate a hand-computed "true delayed
benefit" for both paths on every decision — something no real deployment
has. This phase removes that crutch. Nothing here is drawn from a
probability table; every quality number is the result of actually executing
Python code and comparing its output against a correctness criterion.

## What was built

- **`arbiter_gate/adapters/shadow.py`** — `ShadowExecutionAdapter`. The core
  honesty rule it enforces: an outcome is only ever reported to the gate
  when both paths were genuinely executed (the gate's own `"exploring"` or
  `"probe"` decisions). On `"trusted"` or `"blocked"` decisions, only the
  chosen path runs — the counterfactual is not computed, so no outcome is
  reported, matching `ArbiterGate.report_outcome`'s existing no-op behavior
  rather than inventing a number. 3 new unit tests (`test_shadow_adapter.py`)
  check call counts directly (not just return values) to verify the
  untaken path's function is literally never invoked outside
  exploring/probe. 11/11 tests pass (8 from Phase 0 + 3 new).

- **`arbiter_gate/bench/code_repair_pipeline.py`** — a real, executed
  decision pipeline. Each decision repairs a small synthetic "buggy helper"
  with 1–3 hidden boolean repair parameters and one correct combination,
  unknown to the fixer:
  - `cheap_fix`: returns a fixed default guess. No execution-based checking.
  - `expensive_fix`: actually tries every parameter combination and actually
    runs it against a small "smoke" test set available at commit time,
    keeping the first one that passes.
  - The *true* correctness is only checked later, against a larger "edge"
    test set, after `offset=3` further pipeline steps have elapsed — this
    is the delayed-consequence structure (a regression surfacing several
    commits downstream), and it is measured by literally calling the
    chosen repair and comparing its output to the ground truth on 8 input
    points, not simulated.

- A fourth bucket, `trivial_bug`, was added deliberately: its correct repair
  is usually (85% of instances) already equal to the cheap default. This is
  the case Phase 0's design never tested — a bucket where the *right*
  learned behavior is to stop paying for the expensive path, not just
  "expensive always wins." Phase 0's three original-style buckets
  (`offbyone_bug`, `comparison_bug`, `arithmetic_bug`) all have uniformly
  random correct combinations, so expensive genuinely helps on all three.

## A bug found during this phase, and a design flaw it exposed

**Reproducibility bug (same class as Phase 0's):** none found this time —
`hashlib.sha256` was used from the start for the pseudo-random weighting
function specifically *because* Phase 0 had already shown `hash()` on
strings is per-process-randomized.

**A real design flaw, caught by looking at the actual numbers:** the first
version of the weighting function was *linear* in the test input `x`
(`weight(i, x) = (i + 2) * x`). That made the two "smoke" test points
algebraically redundant — the equation at `x=2` was just `2×` the equation
at `x=1` — so they carried no more information than one point. Running the
dev-seed check showed `always_expensive` hitting *exactly* 1.0 quality with
zero variance, which was the tell: smoke was accidentally fully determining
the correct answer every time, defeating the entire point of having a
separate, larger "edge" test set revealed later. Fixed by hashing `(i, x)`
instead of a linear formula, which removed the accidental redundancy —
confirmed by rerunning the dev check and seeing `always_expensive` quality
drop to a genuine, non-degenerate 0.97–0.98 with nonzero variance.

This is disclosed here rather than quietly fixed and left unmentioned,
per the same standing practice as Phase 0.

## Pre-registered tuning (dev seeds 0–2 only)

The one knob tuned before looking at validation data was
`value_margin` (how much measured quality advantage the expensive path must
clear, net of the fact that it costs more, before it's trusted). At the
library's default (`margin=0.0`), the gate stayed on `trusted → expensive`
for `trivial_bug` because its measured LCB came out at ~0 — technically
non-negative, so "trusted" fired in the literal sense, but this is clearly
not the intended behavior: the mechanism should actively prefer cheap when
the two paths are statistically indistinguishable, not just "not proven
worse." Sweeping `margin ∈ {0.0, 0.03, 0.05, 0.08, 0.10}` on seed 0 showed a
step change at `margin=0.03` (cost drops, quality barely moves) and no
further change through `0.10`. **`margin=0.05`** was picked — comfortably
past the step, nowhere near the top of the flat region — and locked before
touching seeds 10–14.

## Validation (reserved seeds 10–14, touched once, 1200 decisions/seed)

| Policy | Quality (mean ± pop. SD) | Cost (mean ± pop. SD) |
|---|---|---|
| `always_cheap` | 0.4597 ± 0.0044 | 1.0000 ± 0.0000 |
| `static_rule` (hardcoded: expensive only on `arithmetic_bug`/`comparison_bug`) | 0.8353 ± 0.0019 | 3.5000 ± 0.0000 |
| `gate_shadow` (this library, real shadow execution, real delayed reporting) | **0.9615 ± 0.0036** | **3.7982 ± 0.0443** |
| `always_expensive` | 0.9780 ± 0.0038 | 4.0000 ± 0.0000 |

Reading this honestly:

- `gate_shadow` recovers 96.8% of the *available quality gain* over the cheap
  floor ((0.9615−0.4597)/(0.9780−0.4597) = 0.968) at 95.0% of
  `always_expensive`'s cost — it is not a free lunch, but
  it is a real, measured improvement in cost per unit of quality, not an
  artifact of the harness.
- `static_rule` — a plausible policy an engineer might hand-write without
  any learning mechanism — needed slightly *less* cost (3.5 vs 3.80) but
  delivered meaningfully less quality (0.835 vs 0.962). The point of the
  gate is not "cheapest" or "best quality" in isolation; it's a better
  quality-per-cost trade than a static guess, learned from outcomes the
  static rule never sees.
- Per-bucket diagnostics (dev seeds, consistent across 0/1/2) confirm the
  mechanism is doing what it's supposed to, not just defaulting to
  expensive everywhere: measured LCB stays well above the 0.05 margin for
  `offbyone_bug`, `comparison_bug`, and `arithmetic_bug` (0.2–0.6), and
  drops to ~0.0–0.04 for `trivial_bug` — correctly below margin — causing
  the cooldown mechanism to shift `trivial_bug` traffic toward cheap after
  the initial exploration/occasional sparse probes.

## What this does and does not prove

**Does prove:** the ticket/callback + shadow-execution pattern works with a
*real* execution-based signal, not a fed-in number — the gate learns, from
actually running both repair strategies on a bucket, which one the delayed
(not immediate) evidence favors, and that per-bucket learning correctly
differs across buckets with genuinely different cost/benefit profiles,
including a bucket where the right call is "don't bother."

**Does not yet prove:** anything about a real LLM agent. The "cheap" and
"expensive" functions here are still synthetic stand-ins (parameter search
over a toy arithmetic function), chosen because they are *real, executed,
and non-trivial* to game, but they are not an actual agent choosing between
a quick heuristic and a tool call or deep reasoning pass. That integration —
wrapping an actual agent's two code paths (e.g. a coding assistant's
"pattern-match a fix" vs. "run the full test-driven search" decision, with
outcomes measured from real test suites on real repositories) — is the next
step, not yet done here.

## Correction (independent review, 2026-11-19)

An independent review found the quality-gain sentence above previously
misstated as "95.4% of `always_expensive`'s quality" — inconsistent with
its own cited formula, which computes 96.8% of the *available gain*, a
different and correctly-labeled quantity. It also found that
`run_all_seeds()`'s documented CLI reproduction had silently drifted to
the library's default `value_margin` (0.0) instead of the pre-registered
0.05, so the published command did not actually reproduce
`phase1_validation_results.json`. Both are fixed: the sentence above now
matches its own math, and `run_all_seeds()` is pinned to the pre-registered
config — re-verified to reproduce the saved JSON exactly on seeds 10–14.

## Reproducing this

```
cd arbiter_gate
python3 -m pytest tests/ -v
python3 -m arbiter_gate.bench.code_repair_pipeline 10 11 12 13 14
```
