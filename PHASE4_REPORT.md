# Phase 4 report — heterogeneous paired routing on local GPU

Date: 2026-11-19 (Asia/Riyadh)

## What ran

- Real local `qwen2.5:7b` through Ollama on RTX 2070 Max-Q.
- Development seed 0, then reserved validation seeds 10–14 once.
- 30 paired tasks per seed, 150 reserved validation tasks total.
- Both quick and deep were executed for every task; gate deployment was
  simulated on the same paired outcomes, eliminating sample mismatch.
- No free model-generated code was executed. Only fixed candidate letters
  selected pre-written functions.
- CEL-0 was not stopped or modified and advanced to cycle 11,819 on CUDA.
- GPU peak observed during the run left about 3.2GB free.

## Reserved-validation result

| Policy | Quality mean ± SE | Warm latency/task mean ± SE |
|---|---:|---:|
| Always quick | 0.9133 ± 0.0162 | 0.6835s ± 0.0140 |
| ArbiterGate | **0.9567 ± 0.0148** | **0.8559s ± 0.0267** |
| Always deep | 0.9950 ± 0.0020 | 1.0839s ± 0.0127 |

ArbiterGate gained `+0.0433` quality over always-quick and reduced warm
latency by `21.0%` versus always-deep. It occupied a genuine intermediate
quality/latency point rather than merely copying either endpoint.

## Preregistered judgment

The preregistered performance condition required both:

1. quality gain over quick of at least `0.05`; and
2. lower warm latency than always-deep.

Condition 2 passed. Condition 1 narrowly failed (`0.0433 < 0.05`). The correct
verdict is therefore **not promotion**, despite the positive Pareto signal.

Mechanically, the limiting factor was four exploration samples per family.
On seeds 10–12 the lower confidence bound remained at or below zero despite
deep being substantially better over the full paired set. Seeds 13–14 happened
to expose enough useful boundary cases early, so the gate trusted deep there
and achieved larger gains. This order sensitivity is a real limitation, not a
GPU or model failure.

## Honest conclusion

GPU acceleration worked and reduced warm calls to roughly 0.7–1.1 seconds.
The harder task bank successfully created cases where deeper evidence helped.
ArbiterGate produced a useful tradeoff across 150 held-out tasks, but missed
its own predeclared quality threshold. Further tuning must use new seeds and a
new preregistration; seeds 10–14 are now consumed and must not be reused for
selection.

Verdict:
`INTEGRATION_AND_PARETO_SIGNAL_CONFIRMED / PREREGISTERED_PERFORMANCE_GATE_FAILED`.
