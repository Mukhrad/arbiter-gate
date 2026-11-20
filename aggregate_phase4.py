#!/usr/bin/env python3
"""Reusable aggregation for Phase 4's reserved-validation seeds, replacing
the ad-hoc one-off command used when PHASE4_REPORT.md was first produced
(that session hit a usage limit before saving this). Recomputes quality
means from each seed's `*_results.json` records, and warm-latency means
directly from each seed's raw `*_log.jsonl` (summing each task's real
model-call elapsed_s for the deployed path), so the numbers in
PHASE4_REPORT.md can be regenerated and checked, not just trusted.

Usage (from the arbiter_gate folder, after running the 5 validation seeds):
    python aggregate_phase4.py --seeds 10 11 12 13 14
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _quick_and_deep_times(log_path: Path) -> tuple[dict, dict]:
    quick_times, deep_times = {}, {}
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            call = rec.get("call")
            if call == "quick" and "elapsed_s" in rec:
                quick_times[rec["task_id"]] = rec["elapsed_s"]
            elif call in ("deep_attempt1", "deep_attempt2") and "elapsed_s" in rec:
                deep_times.setdefault(rec["task_id"], 0.0)
                deep_times[rec["task_id"]] += rec["elapsed_s"]
    return quick_times, deep_times


def aggregate(seeds: list[int], prefix: str = "phase4_val_seed") -> dict:
    per_seed = []
    for seed in seeds:
        results_path = Path(f"{prefix}{seed}_results.json")
        log_path = Path(f"{prefix}{seed}_log.jsonl")
        with open(results_path, encoding="utf-8") as f:
            records = json.load(f)["records"]
        quick_times, deep_times = _quick_and_deep_times(log_path)

        quick_q = statistics.fmean(r["quick_quality"] for r in records)
        gate_q = statistics.fmean(r["deployed_quality"] for r in records)
        deep_q = statistics.fmean(r["deep_quality"] for r in records)

        gate_seconds, deep_seconds = [], []
        for r in records:
            tid = r["task_id"]
            deep_seconds.append(deep_times.get(tid))
            gate_seconds.append(deep_times.get(tid) if r["use_expensive"] else quick_times.get(tid))

        per_seed.append({
            "seed": seed, "quick": quick_q, "gate": gate_q, "deep": deep_q,
            "gate_seconds": statistics.fmean(gate_seconds),
            "deep_seconds": statistics.fmean(deep_seconds),
        })

    def _mean(key):
        return statistics.fmean(row[key] for row in per_seed)

    def _se(key):
        vals = [row[key] for row in per_seed]
        return statistics.stdev(vals) / (len(vals) ** 0.5) if len(vals) > 1 else 0.0

    summary = {
        "validation_seeds": seeds,
        "per_seed": per_seed,
        "means": {
            "always_quick_quality": _mean("quick"),
            "gate_quality": _mean("gate"),
            "always_deep_quality": _mean("deep"),
            "always_quick_warm_seconds": statistics.fmean(
                statistics.fmean(_quick_and_deep_times(Path(f"{prefix}{s}_log.jsonl"))[0].values())
                for s in seeds
            ),
            "gate_warm_seconds": _mean("gate_seconds"),
            "always_deep_warm_seconds": _mean("deep_seconds"),
        },
    }
    summary["means"]["gate_quality_gain_over_quick"] = (
        summary["means"]["gate_quality"] - summary["means"]["always_quick_quality"]
    )
    summary["means"]["gate_latency_saving_vs_deep"] = (
        1.0 - summary["means"]["gate_warm_seconds"] / summary["means"]["always_deep_warm_seconds"]
    )
    summary["standard_errors"] = {
        "always_quick_quality": _se("quick"),
        "gate_quality": _se("gate"),
        "always_deep_quality": _se("deep"),
        "gate_warm_seconds": _se("gate_seconds"),
        "always_deep_warm_seconds": _se("deep_seconds"),
    }
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12, 13, 14])
    ap.add_argument("--out", default="phase4_aggregate_results.json")
    args = ap.parse_args()
    summary = aggregate(args.seeds)
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["means"], indent=2))
    print(f"\nWrote {args.out}")
