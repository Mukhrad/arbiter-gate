# Phase 5 report — larger exploration budget on GPU (confounded; not a valid performance result)

Date: 2026-11-20 (Asia/Riyadh)

## What ran

- Real local `qwen2.5:7b` through Ollama on GPU, same harness as Phase 4
  plus the two pre-registered Phase 5 changes: `exploratory_decisions=8`
  (library default was 4) and a coded GPU-memory guard (ran clean: 25
  memory checks logged, no abort).
- 5 reserved-seed runs, 30 tasks each (150 total), run from the terminal
  directly rather than through me — I independently reconstructed every
  number below from the raw `phase5_log.jsonl`, since `--out`/`--log`
  weren't given distinct per-seed names and `phase5_results.json` only
  kept the last run's summary. The 5 runs are recovered as 5 contiguous
  150/5=30-task blocks in file order (quick-quality varies meaningfully
  block to block — 0.958, 0.942, 0.917, 0.933, 0.908 — consistent with 5
  distinct seeds' task banks, not one seed repeated).

## Headline numbers (literal pre-registered formula)

| Policy | Quality mean | Warm latency/task mean |
|---|---:|---:|
| Always quick | 0.9317 | 0.702s |
| ArbiterGate | 0.9983 | 1.028s |
| Always deep | 0.9983 | 1.090s |

- Quality gain over quick: **+0.0667** (bar was +0.05 — numerically clears it)
- Latency saving vs. deep: **5.3%** (bar was "lower than deep" — numerically clears it, but much thinner than Phase 4's 21%)

## Why this is not a valid pass, despite clearing both numbers

With `exploratory_decisions=8` and `n_per_bucket=10` unchanged, **24 of 30
decisions per seed (80%) were mandatory "exploring" decisions** — these
always take the expensive/deep path by construction, regardless of
whether deep has any real advantage. Only 6 of 30 decisions per seed
(2 trusted + 4 blocked) reflect the gate actually exercising judgment.

Restricting to only those judgment decisions (30 across all 5 seeds):
gate quality 1.0000 vs. quick 0.9833 vs. deep 1.0000 — a real but tiny
+0.0167 gap, and on the 20 "blocked" decisions the shadow/counterfactual
deep quality was *also* 1.0000, meaning quick and deep were
indistinguishable on exactly the cases the gate chose not to escalate.

The observed +0.0667 "gain" over quick is therefore almost entirely a
byproduct of forcing 80% of decisions onto the expensive path, not
evidence that ArbiterGate's learned routing adds value. Raising
`exploratory_decisions` without raising `n_per_bucket` in proportion
turned this run into an approximation of "always deep, minus a few forced
-quick tasks" rather than a fair test of arbitration. This is my own
design error in Phase 5's configuration (I set exploratory_decisions=8
against the unchanged n_per_bucket=10 from Phase 4), caught only by
checking the reason breakdown after the run rather than before — exactly
the kind of thing this project's own rule (never take a result at face
value, including one's own) exists to catch.

## Honest verdict

`EXECUTION_CLEAN / RESULT_CONFOUNDED_BY_EXPLORATION_FRACTION / NOT_A_VALID_PERFORMANCE_TEST`.

The GPU-memory guard and the harness itself ran correctly (no errors, no
fallback, no CEL-0 impact). But this run cannot be used to support or
refute the preregistered performance claim. Seeds 15–19 are now consumed
on this flawed design and should not be reused for a corrected retest.

## What a valid retest needs

To test a larger exploration budget without this confound, `n_per_bucket`
must grow in proportion to `exploratory_decisions` so post-exploration
("trusted"/"blocked") decisions remain the majority of the sample — e.g.
`exploratory_decisions=8` with `n_per_bucket=40` (20% forced exploration,
80% real arbitration opportunity), using fresh seeds (20+), pre-registered
before any of them is opened.
