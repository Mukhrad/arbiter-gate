"""Run an executable, auditable DevOps routing proof.

Each candidate is copied into a fresh temporary project and graded by a real
pytest process. During exploration the adapter runs both candidates; after
enough measured evidence, Arbiter Gate chooses one path only. No score is
invented for an unexecuted counterfactual.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from arbiter_gate.adapters.shadow import ShadowExecutionAdapter
from arbiter_gate.core.gate import ArbiterGate, ArbiterGateConfig
from arbiter_gate.core.value_estimator import ValueEstimatorConfig


ROOT = Path(__file__).resolve().parent
CASES = (
    ("docs", "docs-only", "Documentation-only change"),
    ("refactor", "mechanical-refactor", "Mechanical refactor"),
    ("auth", "auth-sensitive", "Authentication boundary change"),
    ("migration", "database-migration", "Production database migration"),
)


def run_candidate(case_name: str, candidate_name: str) -> tuple[dict, float]:
    case_dir = ROOT / "cases" / case_name
    with tempfile.TemporaryDirectory(prefix=f"arbiter-{case_name}-") as tmp:
        work = Path(tmp)
        shutil.copy2(case_dir / f"{candidate_name}.py", work / "candidate.py")
        shutil.copy2(case_dir / "test_candidate.py", work / "test_candidate.py")
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "test_candidate.py"],
            cwd=work,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    passed = completed.returncode == 0
    payload = {
        "candidate": candidate_name,
        "passed": passed,
        "exit_code": completed.returncode,
        "test_output": completed.stdout.strip(),
    }
    return payload, 1.0 if passed else 0.0


def run_proof(output_dir: Path, rounds: int = 3) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "decision_audit.jsonl"
    config = ArbiterGateConfig(
        exploratory_decisions=2,
        value_config=ValueEstimatorConfig(
            uncertainty_penalty=0.0,
            min_samples_for_lcb=2,
            value_margin=0.0,
        ),
    )
    gate = ArbiterGate(config)
    audit: list[dict] = []

    for round_index in range(1, rounds + 1):
        for case_name, bucket, title in CASES:
            adapter = ShadowExecutionAdapter(
                gate=gate,
                cheap_fn=lambda n=case_name: run_candidate(n, "cheap"),
                expensive_fn=lambda n=case_name: run_candidate(n, "deep"),
                cheap_cost=1.0,
                expensive_cost=5.0,
            )
            result = adapter.decide_and_run(bucket)
            if result.immediate_advantage is not None:
                adapter.report_delayed(result.decision, result.immediate_advantage)

            row = {
                "round": round_index,
                "case": case_name,
                "title": title,
                "bucket": bucket,
                "decision": {
                    "selected_path": "deep" if result.decision.use_expensive else "cheap",
                    "reason": result.decision.reason,
                },
                "deployed": result.deployed_payload,
                "shadow": result.shadow_payload,
                "measured_advantage_deep_minus_cheap": result.immediate_advantage,
                "deployed_cost_units": result.deployed_cost,
                "shadow_cost_units": result.shadow_cost,
            }
            audit.append(row)

    audit_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in audit),
        encoding="utf-8",
    )
    final_round = [row for row in audit if row["round"] == rounds]
    summary = {
        "verdict": "LIVE_CI_PROOF_PASSED"
        if all(row["deployed"]["passed"] for row in final_round)
        else "LIVE_CI_PROOF_FAILED",
        "rounds": rounds,
        "decisions": len(audit),
        "final_routes": {row["case"]: row["decision"]["selected_path"] for row in final_round},
        "all_final_deployments_passed": all(row["deployed"]["passed"] for row in final_round),
        "gate_diagnostics": gate.diagnostics(),
        "audit_log": audit_path.name,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    summary = run_proof(args.output, args.rounds)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["verdict"] == "LIVE_CI_PROOF_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
