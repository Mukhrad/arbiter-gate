# Arbiter Gate: executable DevOps proof

This example is deliberately narrower than a product claim. It proves that
the released Arbiter Gate code can route between two real candidate paths,
grade both with an actual CI test process during exploration, learn from the
measured difference, and later execute only the selected path.

It does **not** claim that an LLM generated these candidates. The candidates
are deterministic fixtures so anyone can reproduce the routing and audit
logic without an API key or model variance.

## What CI executes

Four code-change patterns are tested for three rounds:

| Pattern | Cheap candidate | Deep candidate | Expected learned route |
|---|---|---|---|
| Documentation | passes | passes | cheap |
| Mechanical refactor | passes | passes | cheap |
| Authentication boundary | fails future-token security test | passes | deep |
| Database migration | fails non-blocking migration test | passes | deep |

The first two rounds execute both paths and report the measured advantage.
The third round uses the learned route. CI fails unless every deployed third-
round candidate passes its real tests.

## Run locally

```bash
python -m pip install -e ".[test]"
python examples/devops_live_proof/run_proof.py --output live-proof-artifacts
```

Outputs:

- `summary.json`: final routes, verdict, and gate diagnostics.
- `decision_audit.jsonl`: one complete record per routing decision, including
  which paths actually ran and the unedited pytest output.

The GitHub Actions workflow uploads both files as the
`arbiter-gate-decision-evidence` artifact on every push and pull request.
