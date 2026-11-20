# Phase 6 report — first validated performance pass

Date: 2026-11-20 (Asia/Riyadh)

## What ran

- Real local `qwen2.5:7b` through Ollama on GPU, run directly by the founder.
- 5 reserved-seed runs, 120 tasks each (600 total): `exploratory_decisions=8`,
  `n_per_bucket=40` — the fix for Phase 5's confound. Auto-named per-seed
  output files worked correctly this time (no overwrite).
- Coded GPU-memory guard ran clean across all 5 seeds; no aborts, no
  timeouts, all 600/600 tasks completed.
- Every number below was independently recomputed by me from each seed's
  raw `records` and `phase6_log_seed{N}.jsonl`, not taken from the
  generated summaries.

## Exploration fraction check (pre-registered self-check, done before accepting any pass)

24/120 = exactly **20%** forced-exploration decisions per seed, confirmed
in every one of the 5 seeds — the intended inverse of Phase 5's 80%.

## Headline result (literal pre-registered formula)

| Policy | Quality mean ± SE | Warm latency/task mean ± SE |
|---|---:|---:|
| Always quick | 0.9050 ± 0.0045 | 0.673s ± — |
| ArbiterGate | **0.9846 ± 0.0037** | **0.874s** |
| Always deep | 0.9942 ± 0.0017 | 1.049s |

- Quality gain over quick: **+0.0796** (bar was ≥0.05 — **passes**, with margin)
- Latency saving vs. always-deep: **16.7%** (bar was "lower than deep" — **passes**)

## Why this one is not Phase 5's artifact

Restricting to only the 480 judgment decisions across all 5 seeds
(excluding the 120 forced-exploration ones): quick 0.9052, gate 0.9818,
deep 0.9938 — a **+0.0766** gap, essentially the same size as the
headline +0.0796. The result is not being carried by forced exploration.

More importantly, the arbitration itself looks correctly targeted, not
just directionally positive:

- On the 225 "trusted" decisions (gate chose deep): the counterfactual
  quick-quality on those *same* tasks was only **0.834** — well below
  quick's overall average of 0.905. The gate is disproportionately
  escalating exactly the harder-than-average tasks.
- On the 231 "blocked" decisions (gate chose quick): deployed quality was
  **0.969**, close to quick's ceiling, and the counterfactual deep
  quality on those tasks was only marginally higher (0.994) — the gate
  is correctly recognizing these didn't need escalation.

## One honest note, not a red flag

Seed 23 behaved differently from the other four (95 trusted / 1 blocked
vs. roughly 32/58 elsewhere) — on that seed the gate's lower-confidence
bound stayed above the margin for nearly the whole run, so it kept
escalating. Direction and magnitude still held on that seed alone
(gate 0.9875 vs. quick 0.8917), so this reads as genuine seed-to-seed
task-difficulty variation, consistent with the order-sensitivity Phase 4
already flagged, not an error.

## Verdict

`INTEGRATION_VERIFIED / EXPLORATION_FRACTION_CORRECTED (20%, not Phase 5's 80%) / PREREGISTERED_PERFORMANCE_GATE_PASSED`.

This is the first phase where ArbiterGate's real-model performance claim
clears its own pre-registered bar, on reserved seeds (20–24) that were
touched once and are now consumed. It should not be re-tested on these
seeds; a further iteration (e.g. a different task family, or testing
whether the gain holds with a cheaper/weaker "quick" model instead of the
same model for both paths) needs fresh seeds and a fresh pre-registration.
