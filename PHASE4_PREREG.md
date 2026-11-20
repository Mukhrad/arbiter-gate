# Phase 4 preregistration — heterogeneous local-LLM routing on GPU

Frozen before any Phase-4 model call.

- Model: `qwen2.5:7b`, same model for quick and deep.
- Compute: Ollama GPU, provided at least 1GB remains free after load; CEL-0 is
  not stopped or modified.
- Development seed: 0, 10 tasks/family.
- Reserved validation seeds: 10–14, 10 tasks/family/seed. No task or threshold
  changes after seed 10 is opened.
- Families remain `boundary_bug`, `arith_bug`, `logic_bug`.
- Heterogeneity fixed within family: hard rates 0.80, 0.50, 0.20 respectively.
- Clear item: quick sees full specification. Hard item: quick sees a deliberately
  incomplete real-world-style request; deep sees full specification and public
  test evidence. Both are graded only by hidden tests.
- Every task is paired: quick and deep are each called once (deep may retry only
  after failing public tests). Gate deployment is simulated on those same paired
  outcomes, eliminating sampling mismatch.
- Baselines: always-quick, always-deep, and gate-deployed quality/cost.
- Integration success: all tasks finish, no unsafe execution/fallback, raw log
  and hashes produced.
- Development-only amendment, frozen before opening seed 10: both paths used
  one model call on seed 0, so call count cannot represent their different
  prompt/inference burden. The cost endpoint is therefore warm inference
  wall-time from the raw Ollama log (the first cold-load call of each process
  is reported separately and excluded from the paired latency comparison).
- Performance evidence (not promotion): across reserved seeds, gate must beat
  always-quick quality by at least 0.05 while its simulated deployed warm
  latency is lower than always-deep. Otherwise the performance claim fails or
  is inconclusive.
- Safety: only fixed candidate letters are accepted; no model-generated code is
  executed. Max run 45 minutes per seed.
