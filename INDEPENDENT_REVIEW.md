# Independent review — 2026-11-19

Reviewed from `arbiter_gate_mvp.zip` (SHA-256
`B2E1DD73EDBCDAFE0A63D14385980E18AEE9CBA55541696B9CA0C967841A9FB1`).

## Reproduced

- Unit suite: 11/11 passed on Python 3.13.1.
- Phase 2 reserved seeds 10–14: deterministic across processes and exact
  structural match with `phase2_validation_results.json`.
- The shared imperfect parser produces the disclosed classifier-gap rates;
  the gate does not receive the hidden true query label.

## Defect found and corrected

The Phase 1 report pre-registered `value_margin=0.05`, but the published CLI
reproduction path used the library default (`0.0`). Its output therefore did
not match `phase1_validation_results.json`. Passing `0.05` reproduced the
saved JSON exactly. `run_all_seeds()` is now pinned to the pre-registered
margin, making the documented CLI command reproducible.

The report's quality-gain sentence was also corrected from an inconsistent
95.4% label to the value shown by its own equation: 96.8% of the available
gain over the cheap floor.

## Production boundary

The guide now distinguishes deployed-path feedback from a true counterfactual
advantage. User ratings and re-asks are valuable outcomes, but cannot score an
answer the user never saw without matched randomization, an independent
evaluator, or another declared causal estimator.

Verdict: `MECHANICS_VERIFIED / TWO_EXECUTABLE_DOMAINS / LIVE_LLM_NOT_YET_RUN`.
