# Security and data handling

## What this library touches

`arbiter_gate` (core/adapters/integrations) is a pure in-process decision
component. It reads only what the caller passes it directly — a bucket
label (a string you choose) and, when you report an outcome, a numeric
advantage value. It does not read files, does not make network calls, and
has zero third-party runtime dependencies.

It never sees your prompts, diffs, source code, or model outputs unless
your own `cheap_fn` / `expensive_fn` callables choose to pass that data
through it — and even then, the library only stores the small bucket
label and the numeric decision/outcome trail described below, not the
payloads themselves.

## What state it keeps, and where

In-memory only, for the lifetime of the `ArbiterGate` / `MultiStepValueEstimator`
object you create: per-bucket running statistics (mean, variance, sample
count) and a small table of open decision tickets awaiting an outcome
(capped by `pending_ttl_seconds`, default 1 hour). Nothing is written to
disk, a database, or any external service by the library itself.

The `examples/devops_live_proof/` driver is a reference integration, not
part of the installed package: it writes a local JSONL decision log and
`summary.json` to whatever `--output` directory you point it at. Review
that script before running it against anything beyond its own bundled
toy cases.

## Secrets and credentials

The library takes no API keys, tokens, or credentials — it has nothing to
authenticate to. If you wire it into a system that does call external
model APIs (via your own `cheap_fn`/`expensive_fn`), those credentials are
yours to manage; arbiter_gate never sees them.

## Reporting a problem

This is a young, Apache-2.0 licensed project (v0.2.1) without a dedicated
security contact yet. Open a GitHub issue for anything other than a
sensitive disclosure; for something you'd rather not post publicly,
contact the maintainer directly via the email on the repository's commit
history.
