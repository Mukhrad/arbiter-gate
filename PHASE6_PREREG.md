# Phase 6 preregistration — corrected exploration-to-arbitration ratio

Frozen before any Phase-6 model call, and before seed 20 is opened.

## What this fixes

Phase 5 raised `exploratory_decisions` from 4 to 8 but left `n_per_bucket`
at 10, so 8/10 = 80% of decisions per bucket were mandatory "exploring"
(always the expensive path by construction). Only 6/30 decisions per seed
reflected the gate's actual judgment, and the reported quality gain was
almost entirely an artifact of that forced-exploration fraction, not
evidence of learned arbitration (see `PHASE5_REPORT.md`).

Phase 6 keeps `exploratory_decisions=8` (unchanged — it still addresses
Phase 4's diagnosed sample-size problem) and raises `n_per_bucket` to
**40**, so the forced-exploration fraction drops to 8/40 = **20%**, leaving
32/40 (80%) of decisions per bucket as real trusted/blocked/probe
judgment — the inverse of Phase 5's ratio. `ArbiterGate` /
`ArbiterGateConfig` remain unmodified; this is an experiment-configuration
fix only.

**Also fixed**: Phase 5's results were nearly lost because 5 seed-runs
were written to the same default `--out`/`--log` filenames, overwriting
the summary each time (only the raw log, being append-only, preserved
everything, by luck). `run_phase6_gpu_experiment.py` now derives
`--out`/`--log` defaults from `--seed` automatically
(`phase6_results_seed{N}.json`, `phase6_log_seed{N}.jsonl`) unless
explicitly overridden, so this class of mistake can't recur silently.

## Everything else is unchanged from Phase 4/5

Same model (`qwen2.5:7b`) for both paths, same task bank generator
(`phase4_task_bank.make_phase4_task_bank`, unmodified) and its fixed
hard-rates (boundary 0.80, arithmetic 0.50, logic 0.20), same paired
design, same safety constraint (only a pre-written candidate letter is
ever executed, in an isolated subprocess, with a timeout), same cost
metric (real warm wall-clock time, cold-load excluded), same coded
GPU-memory guard (checked before the run and every 5 tasks, fails closed,
never touches CEL-0), same two-part performance bar.

## Seeds

- Sanity check only (not a validation touch): seed 0 may be re-run with
  the new `n_per_bucket=40` to confirm the harness still behaves, since
  seed 0 was always designated development-only and carries no
  performance claim either way.
- Reserved validation seeds: **20–24**, 40 tasks/bucket × 3 buckets = 120
  tasks/seed, 600 tasks total, touched once. Seeds 10–14 (Phase 4) and
  15–19 (Phase 5, confounded) are not reused for this decision.

## Performance condition (same bar as Phase 4/5, unchanged)

The gate must beat always-quick quality by at least `0.05` **and** its
simulated deployed warm latency must be lower than always-deep's. Both
must hold for a pass. This time, before accepting a pass, I will also
check the decision-reason breakdown myself (exploring vs. trusted vs.
blocked counts) to confirm the result isn't again dominated by forced
exploration — this check is itself pre-registered as part of how Phase 6
will be judged, not added after seeing the numbers.

## Safety

Only fixed candidate letters are accepted; no model-generated code is
executed. Max run per seed raised to 60 minutes (120 tasks/seed is ~4x
Phase 4/5's per-seed workload). CEL-0 is never stopped or modified.
