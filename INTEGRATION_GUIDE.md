# Integrating Arbiter Gate into a real agent

Everything in `PHASE1_REPORT.md` and `PHASE2_REPORT.md` is real, executed
code with real pass/fail or exact-match grading — but neither demo calls an
actual LLM. This environment has no model API credentials, and attempting
to read them out of the environment to make one would be a
credential-exfiltration action, not a legitimate build step — so that
integration has to happen in *your* environment, with *your* key. This
guide is the exact, minimal wiring to do it, using the same two pieces of
infrastructure validated in Phase 1 and Phase 2 unchanged:
`ArbiterGate` + `ShadowExecutionAdapter`.

## The shape of the decision in a real agent

Almost every agent framework has at least one point that looks like this:

```python
if query_is_simple(query):
    answer = quick_answer(query)          # one model call, no tools
else:
    answer = deep_answer(query)           # tool use / multi-step reasoning / retries
```

`query_is_simple` is usually a cheap heuristic, hand-tuned once and never
revisited. That is exactly the brittle static policy `static_rule` played
in both Phase 1 and Phase 2 — it can be a reasonable guess, but it never
learns from what actually happened, and it can't un-learn a bad guess or
re-trust itself after conditions change.

## Minimal real wiring (Anthropic API example)

```python
from arbiter_gate import ArbiterGate
from arbiter_gate.adapters.shadow import ShadowExecutionAdapter
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the caller's own env
gate = ArbiterGate()

def quick_answer(query: str) -> tuple[str, float]:
    resp = client.messages.create(
        model="YOUR_CHEAP_MODEL", max_tokens=300,
        messages=[{"role": "user", "content": query}],
    )
    text = resp.content[0].text
    quality = grade(query, text)  # YOUR scorer -- see below
    return text, quality

def deep_answer(query: str) -> tuple[str, float]:
    resp = client.messages.create(
        model="YOUR_EXPENSIVE_MODEL", max_tokens=1500, tools=YOUR_TOOLS,
        messages=[{"role": "user", "content": query}],
    )
    text = resp.content[-1].text
    quality = grade(query, text)
    return text, quality

bucket = classify(query)  # your own real context classifier -- see note below
adapter = ShadowExecutionAdapter(
    gate=gate, cheap_fn=lambda: quick_answer(query), expensive_fn=lambda: deep_answer(query),
    cheap_cost=1.0, expensive_cost=5.0,  # whatever cost units you track (tokens, $, latency)
)
run = adapter.decide_and_run(bucket)
# ship run.deployed_payload to the user now.
# ... later, once you actually know how it went ...
if run.immediate_advantage is not None:
    adapter.report_delayed(run.decision, run.immediate_advantage)
```

That's the entire integration surface. Nothing about `ArbiterGate` or
`ShadowExecutionAdapter` changed between the Phase 1 code-repair demo, the
Phase 2 QA-agent demo, and this — which is itself the evidence that this is
reusable infrastructure, not a benchmark-specific trick.

## The one real design decision left to you: `grade()`

Phase 1 and Phase 2 could grade instantly because the ground truth (did the
tests pass, did the answer exact-match) was mechanically checkable. A real
agent answering open-ended queries usually does NOT have that — so
`report_delayed` is deliberately decoupled from `decide_and_run` to let you
plug in whatever real signal you actually have, whenever it actually
arrives:

- **Downstream task success**: if the answer feeds a later action (a
  generated SQL query that actually runs, a generated patch that actually
  passes CI), use that real execution result the moment it's known — this
  is precisely the delayed-credit pattern Phase 1 validated.
- **A periodic human/eval-set review** against gold answers, run on a
  sample — closest to how Phase 2's grading worked, but on your own data.

**Thumbs up/down and "did the user re-ask" are real, useful signals about
the DEPLOYED path — but they are not by themselves a counterfactual
advantage** (correction from independent review, 2026-11-19): they tell
you how the answer the user actually saw landed, not how the unshown
shadow answer would have scored. Using them to directly fill `advantage`
quietly reintroduces exactly the omniscient-harness problem Phase 1 was
built to remove, just with a plausible-sounding one-sided guess standing
in for the missing side. Use them as a real signal only with matched/
randomized traffic, an independently scored shadow answer, or another
explicitly declared causal estimator — not as a stand-in for the untaken
path's score.

Whichever source you pick, only call `report_delayed` with an `advantage`
that came from a comparable scorer evaluating BOTH sides, or from a
pre-declared causal estimator — never from deployed-path feedback alone.
`run.immediate_advantage` is valid precisely because the adapter actually
ran both paths and scored both the same way; a ticket holding only
deployed-path feedback is an outcome about that one path, not yet a
counterfactual advantage, and reporting it as one is the same error Phase 1
was built to rule out.

## The one real risk this doesn't solve for you

Phase 2's honest finding applies here directly: **the gate is only as good
as your bucket classifier.** If your `classify(query)` conflates two
genuinely different query types into one bucket, the gate will learn a
single compromise policy for both, same as Phase 2's `singlehop`/`multihop`
example showed for the 15% of multihop queries an under-specified regex
missed. Investing in a better context classifier is usually higher-leverage
than tuning the gate's own parameters (`value_margin`, `exploratory_decisions`,
`negative_streak_for_cooldown`) — those are provided and documented in
`arbiter_gate/core/gate.py`, `value_estimator.py`, and `cooldown.py`, but
Phase 1 and Phase 2 both needed little to no tuning once the bucketing
itself was sound.
