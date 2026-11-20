from pathlib import Path
import sys


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "devops_live_proof"
sys.path.insert(0, str(EXAMPLE.parent))

from devops_live_proof.run_proof import run_proof  # noqa: E402


def test_live_proof_learns_safe_routes_from_executed_tests(tmp_path):
    summary = run_proof(tmp_path, rounds=3)

    assert summary["verdict"] == "LIVE_CI_PROOF_PASSED"
    assert summary["all_final_deployments_passed"] is True
    assert summary["final_routes"] == {
        "docs": "cheap",
        "refactor": "cheap",
        "auth": "deep",
        "migration": "deep",
    }
    assert (tmp_path / "decision_audit.jsonl").exists()
    assert (tmp_path / "summary.json").exists()
