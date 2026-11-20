# Phase 3 — real local LLM integration (Ollama/Qwen)

Date: 2026-11-19 (Asia/Riyadh)

## Execution

- Model: `qwen2.5:7b`, served locally by Ollama.
- Device forced by every request: CPU (`num_gpu=0`).
- Tasks: 60 safe candidate-selection code-repair tasks, 20 in each of
  `arith_bug`, `boundary_bug`, and `logic_bug`.
- Gate: unmodified `ArbiterGate` + `ShadowExecutionAdapter`.
- Delayed reporting offset: 3 tasks.
- No API key, network service, paid inference, or model-generated code
  execution was used.
- CEL-0 remained active independently on CUDA.

## Mechanical verification

- Automated suite: 15/15 passed on the founder's Windows environment.
- Ollama preflight: successful real response in 22.9 seconds.
- Full run: 60/60 tasks completed without timeout or fallback.
- Raw output hashes:
  - `phase3_results.json`:
    `9A0EDAE2EB0F5810AAF8064BA4DE68A5733DC7782181FE69EBA0BBD9C51BAD5B`
  - `phase3_log.jsonl`:
    `8D1C5EEB8B8A41F85F18F9C9A9BE60D33B5E0747878AE9193C8FFFF1C15FAEFB`

## Results

- Deployed mean hidden-test quality: `0.995833`.
- Deployed exact candidate accuracy: `59/60 = 98.33%`.
- Decision reasons: 12 exploring, 3 probe, 45 blocked, 0 trusted.
- Quick path: 60 calls, mean hidden-test quality `1.0`.
- Deep path: 15 calls, mean hidden-test quality `0.983333`.
- Paired shadow comparisons: 15; 14 ties and one case where deep was worse
  by `0.25`; no case where deep was better.
- Total real model calls: 75. The deployed-cost field in the generated
  summary is `1.0` per task, but the experimental cost including shadow
  exploration is `75/60 = 1.25` model calls per task.
- Timed inference: 75 calls, 987.78 seconds total model time, 13.17 seconds
  mean, 31.89 seconds maximum.

## Honest interpretation

The real-model integration works. The gate received real paired evidence,
reported it after a real offset, learned that the deep path had no measured
advantage in any bucket, and shifted 45/60 decisions to the quick path.

This is **not evidence that ArbiterGate improves Qwen quality or cost on this
task bank**. Qwen's quick path was already perfect on hidden-test quality, so
the deep path had no upside to discover; exploration caused the run's only
wrong exact label. An always-quick policy would have been both cheaper and
better on these 60 tasks.

The high aggregate score therefore validates the transport, safety,
shadow-execution, delayed-credit, and fallback mechanics—not commercial
superiority. The next performance experiment needs pre-registered tasks with
genuine within-bucket heterogeneity: some instances where public evidence or
a second attempt helps, and others where it does not. Reserved seeds and an
always-quick baseline must be fixed before that run.

Verdict:
`LIVE_LOCAL_LLM_INTEGRATION_VERIFIED / GATE_BEHAVIOR_CORRECT / PERFORMANCE_ADVANTAGE_NOT_DEMONSTRATED`.
