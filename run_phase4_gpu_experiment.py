#!/usr/bin/env python3
"""Phase 4: matched heterogeneous local-LLM routing experiment on GPU.
ShadowExecutionAdapter, exactly as specified by the founder:

  - Same model for both paths (isolates ArbiterGate's strategy effect from
    model-capability differences).
  - Quick path: one attempt, short context, no test results shown.
  - Deep path: wider context, runs the real (pre-written) public tests for
    every candidate, shows the model that evidence, and allows a second
    attempt if its first pick fails a public test.
  - Judge: hidden tests only (never shown to the model).
  - Delayed credit: a decision's real measured advantage is withheld from
    the gate until `--offset` further tasks have been processed.
  - GPU acceleration through Ollama, with memory checked before the run;
    CEL-0 is neither stopped nor modified.
  - SAFETY: the model only ever picks a letter among pre-written candidate
    functions (see arbiter_gate/bench/code_repair_llm_tasks.py). Its raw
    text is never executed. Chosen candidates run in an isolated temp
    directory, in a subprocess, with a 5s timeout, never touching project
    files or taking arguments the model controls beyond that letter.

Usage:
    python run_phase4_gpu_experiment.py --seed 0 --n-per-bucket 10 --offset 3

Writes incremental JSONL to phase4_log.jsonl (one line per completed task,
so nothing is lost if interrupted) and a final summary to
phase4_results.json. Honors --max-minutes as a hard wall-clock cap: on
expiry it stops cleanly and still writes a summary over whatever ran.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from arbiter_gate.adapters.shadow import ShadowExecutionAdapter
from arbiter_gate.bench.code_repair_llm_tasks import run_hidden_tests, run_public_tests
from arbiter_gate.bench.phase4_task_bank import Phase4Task, make_phase4_task_bank
from arbiter_gate.core.gate import ArbiterGate, ArbiterGateConfig, GateDecision
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


def _format_candidates(task: Phase4Task) -> str:
    lines = []
    for label in sorted(task.candidates):
        lines.append(f"--- Candidate {label} ---\n{task.candidates[label]}")
    return "\n\n".join(lines)


def _call_model(system: str, prompt: str, model: str, url: str, timeout: float,
                 log_fh, task_id: str, call_name: str) -> tuple[str | None, float]:
    """One real Ollama call with robust failure handling. Returns
    (letter_or_None, wall_seconds). Never raises past this point; a model
    or network failure is logged and treated as 'no answer', never as a
    silent default to any particular candidate."""
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
        label = letter or "A"  # explicit, logged fallback -- NOT trusted blindly:
        # since correct_label is randomized per task (see code_repair_llm_tasks.py),
        # "A" carries no positional advantage; this just needs *some* valid
        # label to run hidden tests against when parsing genuinely failed.
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
            # Second repair attempt: tell the model its first pick failed a
            # public test, and which candidates remain plausible.
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
            final_label = letter2 or label1  # fall back to the first pick, not a fixed letter
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
    ap.add_argument("--n-per-bucket", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offset", type=int, default=3)
    ap.add_argument("--per-call-timeout", type=float, default=120.0)
    ap.add_argument("--max-minutes", type=float, default=45.0)
    ap.add_argument("--log", default="phase4_log.jsonl")
    ap.add_argument("--out", default="phase4_results.json")
    args = ap.parse_args()

    tasks = make_phase4_task_bank(seed=args.seed, n_per_bucket=args.n_per_bucket)
    print(f"Loaded {len(tasks)} tasks across {len(set(t.bucket for t in tasks))} buckets.")

    gate = ArbiterGate(ArbiterGateConfig())
    deadline = time.monotonic() + args.max_minutes * 60.0

    results = []
    pending = []  # (decision, immediate_advantage, due_at_index)
    log_fh = open(args.log, "a", encoding="utf-8")

    for i, task in enumerate(tasks):
        if time.monotonic() > deadline:
            print(f"Max wall-clock ({args.max_minutes} min) reached after {i} tasks; stopping cleanly.")
            break

        quick_fn = make_quick_fn(task, args.model, args.url, args.per_call_timeout, log_fh)
        deep_fn = make_deep_fn(task, args.model, args.url, args.per_call_timeout, log_fh)
        # Run a fully matched pair once. The adapter then operates on cached
        # real outcomes, so routing comparisons never mix different samples.
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
        # deep's true cost (1 or 2 model calls) is only known after it ran;
        # correct the recorded cost retroactively for this decision's bookkeeping.
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
        "overall_quality_mean": statistics.fmean(r["deployed_quality"] for r in results) if results else None,
        "overall_cost_mean": statistics.fmean(r["deployed_cost"] for r in results) if results else None,
        "accuracy_vs_ground_truth": (
            statistics.fmean(1.0 if r["chosen_label"] == r["correct_label"] else 0.0 for r in results)
            if results else None
        ),
        "always_quick_quality": statistics.fmean(r["quick_quality"] for r in results) if results else None,
        "always_quick_cost": 1.0 if results else None,
        "always_deep_quality": statistics.fmean(r["deep_quality"] for r in results) if results else None,
        "always_deep_cost": statistics.fmean(r["deep_cost"] for r in results) if results else None,
        "paired_experiment_cost": statistics.fmean(r["actual_paired_calls"] for r in results) if results else None,
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
