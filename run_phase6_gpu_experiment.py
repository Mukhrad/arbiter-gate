#!/usr/bin/env python3
"""Phase 6: same paired heterogeneous GPU design as Phase 4/5, fixing the
confound that invalidated Phase 5 (see PHASE5_REPORT.md / PHASE6_PREREG.md).

Phase 5 raised `exploratory_decisions` to 8 but left `n_per_bucket=10`, so
8/10 = 80% of decisions were mandatory "exploring" (always the expensive
path by construction) -- the apparent quality gain was mostly that forced
fraction, not learned arbitration. Phase 6 keeps `exploratory_decisions=8`
(still addresses Phase 4's real sample-size problem) and raises
`n_per_bucket` to 40 by default, so the forced-exploration fraction drops
to 8/40 = 20%, leaving 80% of decisions as real trusted/blocked/probe
judgment. `ArbiterGate` / `ArbiterGateConfig` remain unmodified -- this is
an experiment-configuration fix only.

Also fixed: Phase 5's per-seed summaries were nearly lost because 5 runs
wrote to the same default --out/--log filenames, overwriting the summary
each time (only the raw log survived, being append-only, by luck). Here
--out/--log default to filenames derived from --seed
(phase6_results_seed{N}.json / phase6_log_seed{N}.jsonl) unless explicitly
overridden, so that mistake can't recur silently.

Still a real, coded GPU-memory guard (unchanged from Phase 5): checked
before each seed starts and every --memory-check-every tasks during the
run; if free GPU memory drops below --min-free-mb, or the check itself
fails, the run stops cleanly (partial summary still written) -- it never
touches or stops CEL-0, it only refuses to keep calling Ollama on the GPU.

Everything else is unchanged from Phase 4/5: same model for both paths,
same quick/deep prompts and candidate-letter-only execution, same paired
design (quick and deep each called once per task, gate decision simulated
on the cached pair), same safety constraint (no model-generated code is
ever executed -- only a pre-written candidate letter, run in an isolated
subprocess with a timeout).

Usage (run once per reserved seed, 20-24; --out/--log are auto-named per seed):
    python run_phase6_gpu_experiment.py --seed 20
    python run_phase6_gpu_experiment.py --seed 21
    ... through --seed 24
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

from arbiter_gate.adapters.shadow import ShadowExecutionAdapter
from arbiter_gate.bench.code_repair_llm_tasks import run_hidden_tests, run_public_tests
from arbiter_gate.bench.phase4_task_bank import Phase4Task, make_phase4_task_bank
from arbiter_gate.core.gate import ArbiterGate, ArbiterGateConfig
from arbiter_gate.integrations.ollama_client import OllamaError, chat, parse_letter_choice

QUICK_SYSTEM = (
    "You are a careful code reviewer. You will be shown a specification and "
    "several candidate Python function implementations, exactly one of "
    "which is correct. Reply with your reasoning in at most 2 short "
    "sentences, then end your reply with exactly one line: 'Answer: X' "
    "where X is the letter of the candidate you believe is correct."
)

DEEP_SYSTEM = (
    "You are a careful code reviewer with access to real test results. You "
    "will be shown a specification, several candidate Python function "
    "implementations (exactly one correct), and the results of running "
    "some tests against each candidate. Use the test results as evidence. "
    "Reply with your reasoning in at most 3 short sentences, then end your "
    "reply with exactly one line: 'Answer: X'."
)


def check_gpu_memory_free(min_mb: float) -> tuple[bool, float | None, str]:
    """Real, coded GPU-memory guard (replaces Phase 4's manual nvidia-smi
    monitoring). Returns (ok, free_mb_or_None, raw_output_or_error).
    `ok` is True only when nvidia-smi ran successfully AND free memory is
    at or above min_mb. On any failure to query (no nvidia-smi, no GPU,
    parse error) this returns ok=False -- fail closed, never assume memory
    is fine when we can't verify it."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10.0, check=True,
        )
    except Exception as e:  # FileNotFoundError, CalledProcessError, TimeoutExpired, ...
        return False, None, f"nvidia-smi query failed: {e}"
    text = out.stdout.strip()
    m = re.search(r"(\d+)", text)
    if not m:
        return False, None, f"could not parse nvidia-smi output: {text!r}"
    free_mb = float(m.group(1))
    return free_mb >= min_mb, free_mb, text


def _format_candidates(task: Phase4Task) -> str:
    lines = []
    for label in sorted(task.candidates):
        lines.append(f"--- Candidate {label} ---\n{task.candidates[label]}")
    return "\n\n".join(lines)


def _call_model(system: str, prompt: str, model: str, url: str, timeout: float,
                 log_fh, task_id: str, call_name: str) -> tuple[str | None, float]:
    try:
        reply, elapsed = chat(prompt, model=model, system=system, url=url,
                               timeout=timeout, force_cpu=False)
    except OllamaError as e:
        log_fh.write(json.dumps({"task_id": task_id, "call": call_name, "error": str(e)}) + "\n")
        log_fh.flush()
        return None, 0.0
    letter = parse_letter_choice(reply, list("ABCD"))
    log_fh.write(json.dumps({
        "task_id": task_id, "call": call_name, "elapsed_s": round(elapsed, 2),
        "reply": reply[:500], "parsed_letter": letter,
    }) + "\n")
    log_fh.flush()
    return letter, elapsed


def make_quick_fn(task: Phase4Task, model: str, url: str, timeout: float, log_fh):
    def quick() -> tuple[str, float]:
        prompt = (
            f"Specification:\n{task.quick_spec_text}\n\n"
            f"Candidates:\n{_format_candidates(task)}\n\n"
            f"Which candidate is correct? Answer: X"
        )
        letter, elapsed = _call_model(QUICK_SYSTEM, prompt, model, url, timeout,
                                       log_fh, task.task_id, "quick")
        label = letter or "A"
        used_fallback = letter is None
        quality, _ = run_hidden_tests(task, label)
        log_fh.write(json.dumps({
            "task_id": task.task_id, "call": "quick_result", "chosen": label,
            "used_fallback": used_fallback, "quality": quality,
        }) + "\n")
        log_fh.flush()
        return label, quality
    return quick


def make_deep_fn(task: Phase4Task, model: str, url: str, timeout: float, log_fh):
    def deep() -> tuple[str, float]:
        public_results = {}
        for label in sorted(task.candidates):
            frac, detail = run_public_tests(task, label)
            public_results[label] = (frac, detail)
        evidence = "\n".join(
            f"Candidate {l}: {public_results[l][1]}" for l in sorted(public_results)
        )
        prompt1 = (
            f"Specification:\n{task.spec_text}\n\n"
            f"Candidates:\n{_format_candidates(task)}\n\n"
            f"Public test results (some candidates may coincidentally pass "
            f"these without being fully correct):\n{evidence}\n\n"
            f"Which candidate is correct? Answer: X"
        )
        letter1, _ = _call_model(DEEP_SYSTEM, prompt1, model, url, timeout,
                                  log_fh, task.task_id, "deep_attempt1")
        label1 = letter1 or "A"
        n_calls = 1
        final_label = label1
        pass_frac, _ = public_results.get(label1, (0.0, ""))
        if pass_frac < 1.0:
            remaining = [l for l in task.candidates if l != label1]
            prompt2 = (
                f"Specification:\n{task.spec_text}\n\n"
                f"Candidates:\n{_format_candidates(task)}\n\n"
                f"Public test results:\n{evidence}\n\n"
                f"Your previous answer was Candidate {label1}, but it did not "
                f"pass all public tests ({public_results[label1][1]}). "
                f"Reconsider among the remaining candidates "
                f"({', '.join(remaining)}) and give your final answer. Answer: X"
            )
            letter2, _ = _call_model(DEEP_SYSTEM, prompt2, model, url, timeout,
                                      log_fh, task.task_id, "deep_attempt2")
            n_calls = 2
            final_label = letter2 or label1
        quality, _ = run_hidden_tests(task, final_label)
        log_fh.write(json.dumps({
            "task_id": task.task_id, "call": "deep_result", "chosen": final_label,
            "n_calls": n_calls, "quality": quality,
        }) + "\n")
        log_fh.flush()
        return final_label, quality, n_calls
    def deep_wrapped() -> tuple[str, float]:
        label, quality, n_calls = deep()
        deep_wrapped.last_cost = float(n_calls)
        return label, quality
    deep_wrapped.last_cost = 1.0
    return deep_wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--n-per-bucket", type=int, default=40,
                     help="Raised from Phase 4/5's 10 to 40 so that, with "
                          "exploratory_decisions=8, forced exploration is only "
                          "20%% of decisions per bucket instead of Phase 5's 80%%.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offset", type=int, default=3)
    ap.add_argument("--exploratory-decisions", type=int, default=8,
                     help="Per-bucket exploration budget before ArbiterGate trusts "
                          "any estimate. Unchanged from Phase 5 -- it still addresses "
                          "Phase 4's diagnosed sample-size problem; what changed is "
                          "n_per_bucket, so this fraction is no longer dominant.")
    ap.add_argument("--per-call-timeout", type=float, default=120.0)
    ap.add_argument("--max-minutes", type=float, default=60.0,
                     help="Raised from 45 to 60: n_per_bucket=40 means ~4x Phase 4/5's "
                          "per-seed workload.")
    ap.add_argument("--min-free-mb", type=float, default=1024.0,
                     help="Abort cleanly (never stopping/modifying CEL-0 itself) if "
                          "free GPU memory drops below this.")
    ap.add_argument("--memory-check-every", type=int, default=5,
                     help="Re-check GPU memory every N tasks during the run.")
    ap.add_argument("--log", default=None,
                     help="Defaults to phase6_log_seed{seed}.jsonl if not given -- "
                          "auto-named per seed so Phase 5's overwrite mistake (5 runs "
                          "sharing one default filename) can't recur silently.")
    ap.add_argument("--out", default=None,
                     help="Defaults to phase6_results_seed{seed}.json if not given.")
    args = ap.parse_args()
    if args.log is None:
        args.log = f"phase6_log_seed{args.seed}.jsonl"
    if args.out is None:
        args.out = f"phase6_results_seed{args.seed}.json"
    print(f"Writing log to {args.log} and summary to {args.out}")

    ok, free_mb, raw = check_gpu_memory_free(args.min_free_mb)
    print(f"GPU memory check before run: free={free_mb} MB (raw: {raw!r})")
    if not ok:
        print(f"Aborting before any model call: free GPU memory ({free_mb} MB) is "
              f"below --min-free-mb ({args.min_free_mb} MB), or the memory check "
              f"itself failed. CEL-0 is not touched by this abort.")
        return

    tasks = make_phase4_task_bank(seed=args.seed, n_per_bucket=args.n_per_bucket)
    print(f"Loaded {len(tasks)} tasks across {len(set(t.bucket for t in tasks))} buckets.")

    gate = ArbiterGate(ArbiterGateConfig(exploratory_decisions=args.exploratory_decisions))
    deadline = time.monotonic() + args.max_minutes * 60.0

    results = []
    pending = []
    log_fh = open(args.log, "a", encoding="utf-8")

    for i, task in enumerate(tasks):
        if time.monotonic() > deadline:
            print(f"Max wall-clock ({args.max_minutes} min) reached after {i} tasks; stopping cleanly.")
            break

        if i > 0 and i % args.memory_check_every == 0:
            ok, free_mb, raw = check_gpu_memory_free(args.min_free_mb)
            log_fh.write(json.dumps({"call": "memory_check", "index": i, "free_mb": free_mb, "ok": ok}) + "\n")
            log_fh.flush()
            if not ok:
                print(f"Stopping after {i} tasks: free GPU memory ({free_mb} MB) dropped "
                      f"below --min-free-mb ({args.min_free_mb} MB). CEL-0 is not touched.")
                break

        quick_fn = make_quick_fn(task, args.model, args.url, args.per_call_timeout, log_fh)
        deep_fn = make_deep_fn(task, args.model, args.url, args.per_call_timeout, log_fh)
        quick_label, quick_quality = quick_fn()
        deep_label, deep_quality = deep_fn()
        actual_deep_cost = getattr(deep_fn, "last_cost", 1.0)
        adapter = ShadowExecutionAdapter(
            gate=gate,
            cheap_fn=lambda: (quick_label, quick_quality),
            expensive_fn=lambda: (deep_label, deep_quality),
            cheap_cost=1.0, expensive_cost=getattr(deep_fn, "last_cost", 2.0),
        )
        run = adapter.decide_and_run(task.bucket)
        deployed_cost = actual_deep_cost if run.decision.use_expensive else 1.0

        label_chosen = run.deployed_payload
        quality, _ = run_hidden_tests(task, label_chosen)

        record = {
            "index": i, "task_id": task.task_id, "bucket": task.bucket,
            "reason": run.decision.reason, "use_expensive": run.decision.use_expensive,
            "chosen_label": label_chosen, "correct_label": task.correct_label,
            "deployed_quality": quality, "deployed_cost": deployed_cost,
            "shadow_ran": run.shadow_payload is not None,
            "difficulty": task.difficulty,
            "quick_label": quick_label, "quick_quality": quick_quality,
            "deep_label": deep_label, "deep_quality": deep_quality,
            "deep_cost": actual_deep_cost,
            "actual_paired_calls": 1.0 + actual_deep_cost,
        }
        results.append(record)
        log_fh.write(json.dumps({"task_id": task.task_id, "call": "decision", **record}) + "\n")
        log_fh.flush()
        print(f"[{i+1}/{len(tasks)}] {task.task_id} ({task.bucket}) reason={run.decision.reason} "
              f"chosen={label_chosen} correct={task.correct_label} quality={quality:.2f}")

        if run.immediate_advantage is not None:
            pending.append((run.decision, run.immediate_advantage, i + args.offset))

        still_pending = []
        for decision, advantage, due_at in pending:
            if i >= due_at:
                gate.report_outcome(decision, advantage)
            else:
                still_pending.append((decision, advantage, due_at))
        pending = still_pending

    for decision, advantage, _ in pending:
        gate.report_outcome(decision, advantage)

    log_fh.close()

    summary = {
        "n_tasks_run": len(results),
        "n_tasks_total": len(tasks),
        "model": args.model,
        "offset": args.offset,
        "exploratory_decisions": args.exploratory_decisions,
        "n_per_bucket": args.n_per_bucket,
        "reason_counts": {
            reason: sum(1 for r in results if r["reason"] == reason)
            for reason in ("exploring", "trusted", "blocked", "probe")
        } if results else None,
        "exploration_fraction": (
            sum(1 for r in results if r["reason"] == "exploring") / len(results)
            if results else None
        ),
        "overall_quality_mean": statistics.fmean(r["deployed_quality"] for r in results) if results else None,
        "overall_cost_mean": statistics.fmean(r["deployed_cost"] for r in results) if results else None,
        "accuracy_vs_ground_truth": (
            statistics.fmean(1.0 if r["chosen_label"] == r["correct_label"] else 0.0 for r in results)
            if results else None
        ),
        "always_quick_quality": statistics.fmean(r["quick_quality"] for r in results) if results else None,
        "always_deep_quality": statistics.fmean(r["deep_quality"] for r in results) if results else None,
        "always_deep_cost": statistics.fmean(r["deep_cost"] for r in results) if results else None,
        "per_bucket": {},
        "diagnostics": gate.diagnostics(),
        "records": results,
    }
    for bucket in sorted(set(r["bucket"] for r in results)):
        rows = [r for r in results if r["bucket"] == bucket]
        summary["per_bucket"][bucket] = {
            "n": len(rows),
            "quality_mean": statistics.fmean(r["deployed_quality"] for r in rows),
            "cost_mean": statistics.fmean(r["deployed_cost"] for r in rows),
            "n_exploring_or_probe": sum(1 for r in rows if r["reason"] in ("exploring", "probe")),
            "n_trusted": sum(1 for r in rows if r["reason"] == "trusted"),
            "n_blocked": sum(1 for r in rows if r["reason"] == "blocked"),
        }

    Path(args.out).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote summary to {args.out} and incremental log to {args.log}")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2, default=str))


if __name__ == "__main__":
    main()
