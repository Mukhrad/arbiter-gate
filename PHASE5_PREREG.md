# Phase 5 preregistration — larger exploration budget, coded memory guard

Frozen before any Phase-5 model call, and before seed 15 is opened.

## What changes from Phase 4 (and only this)

1. **Exploration budget.** Phase 4's own diagnosis: on seeds 10–12 the
   gate's lower confidence bound stayed at or below zero even where deep
   was genuinely better, purely because `exploratory_decisions=4` (the
   `ArbiterGate` library default) gave too few paired samples per bucket.
   `ArbiterGate` / `ArbiterGateConfig` are **not modified** — the field was
   already a constructor argument — so this is an experiment-configuration
   change only. Phase 5 fixes `exploratory_decisions=8` (double Phase 4's)
   for every bucket, in every seed below, before any of them is opened.
2. **Coded GPU-memory guard**, replacing Phase 4's manual `nvidia-smi`
   monitoring. `run_phase5_gpu_experiment.py` checks real free GPU memory
   via `nvidia-smi` before the run starts and every 5 tasks during it; if
   free memory drops below 1GB, or the check itself fails for any reason,
   the run stops cleanly with a partial summary written. It never touches
   or stops CEL-0 — it only refuses to keep issuing Ollama GPU calls.

Everything else is identical to Phase 4: same model (`qwen2.5:7b`) for both
paths, same task bank generator (`phase4_task_bank.make_phase4_task_bank`,
unmodified) and its fixed hard-rates (boundary 0.80, arithmetic 0.50, logic
0.20), same paired design (quick and deep each called once per task, the
gate's routing decision simulated on the cached pair), same safety
constraint (only a pre-written candidate letter is ever executed, in an
isolated subprocess, with a timeout — never model-generated code), same
cost metric (real warm-inference wall-clock time, cold-load call excluded),
and the same two-part performance bar.

## Seeds

- Development seed: none needed — the harness itself is unchanged except
  for the two items above, both already exercised in Phase 4's own run and
  in this sandbox's own fail-closed test of the memory guard.
- Reserved validation seeds: **15–19**, 30 tasks each (150 total), touched
  once. Seeds 10–14 are Phase 4's and are not reused for this decision.

## Performance condition (same bar as Phase 4, unchanged)

The gate must beat always-quick quality by at least `0.05` **and** its
simulated deployed warm latency must be lower than always-deep's warm
latency. Both conditions must hold for the result to be reported as a
pass. Otherwise: not promotion, reported honestly either way.

## Safety

Only fixed candidate letters are accepted; no model-generated code is
executed. Max run 45 minutes per seed. CEL-0 is never stopped or modified;
the new memory guard adds a second, automated line of defense on top of
the standing rule, in place of manual observation.
