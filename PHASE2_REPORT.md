# Phase 2: the same infrastructure, a second independent real-execution domain

## Why this exists

The founder's instruction was to integrate the mechanism "at the highest
level" ("ادمج على أعلى مستوى"). The highest-leverage way to demonstrate real
integration readiness without a live model API in this sandbox (there are
no model credentials in this environment, and reading them out to call one
anyway would be a credential-exfiltration action — not something this
project does regardless of who's asking) is to prove the *exact same*
`ArbiterGate` + `ShadowExecutionAdapter` code, unmodified, works as general
agent infrastructure on a second, independent, real-execution domain — not
just the code-repair pipeline it was built against in Phase 1. If the
adapter needed domain-specific changes to work here, it would mean Phase 1
accidentally validated a benchmark trick, not a library. It didn't need any.

## The domain: a real multi-hop tool-using QA agent

A small synthetic knowledge base (30 entities, each with its own value and
a link to another entity). Two query shapes:
- **singlehop**: "What is the value of E7?" — needs exactly one real
  dictionary lookup (the "tool call").
- **multihop**: "What is the value of the entity linked to E7?" — needs
  two real, chained lookups: first resolve what E7 links to, then look
  that entity's value up.

This is a real instance of the single-most-common "cheap vs. expensive"
decision point in agent infrastructure today: answer from one tool call,
or chain multiple tool calls / do multi-step reasoning. Two real,
executable strategies:

- `quick_step`: always exactly one lookup. Right on singlehop, silently
  WRONG on multihop (a real, common agent failure — mistaking a chained
  question for a direct one).
- `deep_step`: actually parses the query and performs the second real
  lookup when needed. Right on both query types — *unless* its own parsing
  misses the query's structure (see below).

A real regex-based classifier (`classify_bucket`) assigns the context
bucket the gate uses — not a hidden ground-truth label. `deep_step` shares
the *exact same* regex to decide whether to chain. That sharing is
deliberate: in a real system the router and the agent's own intent-parsing
are often the same or closely related code, so a gap in that parsing is a
gap for everyone, not something a "smarter" path magically sees past.

## The honest limitation, found and disclosed, not fixed

~15% of multihop queries use an alternate phrasing ("E7's successor's
value?") the regex doesn't recognize. The first version of this benchmark
accidentally had the classifier recognize *both* phrasings (a second regex
covered the alternate one too), which meant nothing in the benchmark could
ever actually fail from a classification gap — an unrealistic benchmark
that would have hidden exactly the kind of risk `INTEGRATION_GUIDE.md` now
calls out as the main real risk this library doesn't solve for you. Caught
by checking the `_classifier_gap` diagnostic (which exists specifically to
surface this, not to make the result look better) and reading 0.0% as the
tell that something was wrong, not as a win. Fixed by making `deep_step`
and the bucket classifier share one parser (`_looks_multihop`) that only
recognizes the canonical phrasing — so the alternate phrasing is now a
real, measured blind spot, at a real, reproducible rate (~14–15% across
seeds), affecting **every** strategy that depends on parsing the query,
not just the cheap one.

## Validation (reserved seeds 10–14, touched once, 1200 queries/seed)

| Policy | Quality (mean ± pop. SD) | Cost (mean ± pop. SD) |
|---|---|---|
| `always_quick` | 0.5072 ± 0.0173 | 1.0000 ± 0.0000 |
| `static_rule` (hardcoded: chain only when classifier says multihop) | 0.9308 ± 0.0061 | 1.4385 ± 0.0104 |
| `gate_shadow` (ArbiterGate, unmodified from Phase 1, real shadow execution) | **0.9307 ± 0.0060** | **1.4383 ± 0.0104** |
| `always_deep` | 0.9308 ± 0.0061 | 1.4385 ± 0.0104 |

Mean real classifier-gap rate across the validation seeds: **14.2%** of
true multihop queries silently bucketed as singlehop — this caps every
policy's achievable quality at ≈0.93, not 1.0, and is disclosed rather than
smoothed over.

## Reading this honestly

The gate essentially **reproduces** `static_rule`'s cost/quality point
here, to three decimal places — it did not beat the hand-coded policy.
That is a materially different, and more modest, result than Phase 1's
(where the gate clearly beat the static rule). The honest reason: in this
domain, within each bucket the two paths' quality difference is exactly
constant (0 inside `singlehop`, 1.0 inside `multihop` — no per-instance
noise), so there was nothing left for a learning mechanism to discover
that a one-time hand-coded rule, written by someone who'd thought about the
buckets, couldn't already get right. **The real win here is not "beats the
static rule" — it's "needed zero hand-tuned rules and no knowledge of which
bucket matters, and still converged, from real delayed measurement alone,
to the same operating point a human had to sit down and hand-derive."**
That is a meaningfully weaker claim than Phase 1's and is stated as such,
not inflated.

Where the gate's advantage over a static rule actually shows up is the
situation Phase 1 demonstrated and this domain doesn't: buckets with
genuine *per-instance* uncertainty (sometimes the expensive path helps,
sometimes it doesn't, and not predictably from the bucket label alone), and
conditions that drift over time so a rule written once goes stale while a
measuring mechanism keeps re-checking. This domain's buckets are too clean
for that effect to appear — which is itself a useful, disclosed boundary
on when this mechanism earns its complexity over a simple static rule.

## What this phase proves and does not prove

**Proves:** `ArbiterGate` + `ShadowExecutionAdapter`, with literally no
code changes, functioned correctly as the control layer for a second,
independent, real-execution agentic domain (tool-call chaining) distinct
from Phase 1's (code repair) — real lookups, real exact-match grading, real
delayed reporting, real (and disclosed) classifier limitations. This is
the evidence that the Phase 0 plan's core claim — this is reusable
infrastructure, not a one-off mechanism — survives contact with a second
domain.

**Does not prove:** that it beats a well-written static rule in every
domain (it didn't, here, and that's reported rather than hidden), or
anything about a real LLM-backed agent — no model API call was made
anywhere in this project; `INTEGRATION_GUIDE.md` is the literal, unexecuted
wiring for the user's own environment and credentials, not a run that
happened here.

## Reproducing this

```
cd arbiter_gate
python3 -m pytest tests/ -v
python3 -m arbiter_gate.bench.multihop_qa_agent 10 11 12 13 14
```
