# Agent Evaluation Harness

[![CI](https://github.com/Ruveshan/agent-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/Ruveshan/agent-eval-harness/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A Python framework that answers one question: **is this LLM agent reliable
enough to put in front of real money?**

It ships with a demo agent — a bank-transaction classifier that runs on
**Google Gemini's free API tier** by default (get a key at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey), no card
required). An Anthropic Claude implementation of the same agent is included
behind `--agent claude`, because the harness evaluates *any* agent that
implements a three-member interface. The output is a CI-gating exit code, a
terminal scorecard, `results.json` + `report.md`, and a Streamlit dashboard.

## Why agent reliability matters in payments

A transaction classifier that is 95% accurate sounds good until you do the
arithmetic on a million transactions a day. And accuracy is only one of four
independent ways an LLM agent fails in production:

| Failure mode | What it looks like | What it costs |
|---|---|---|
| **Wrong answer** | "UBER EATS" classified as transport | Bad budgets, misfired fraud rules, support tickets |
| **Inconsistency** | Same input, different answer on Tuesday | Un-debuggable behavior, flaky downstream tests |
| **Schema drift** | Prose apology instead of JSON | Pipeline crashes — the worst failure is the unparseable one |
| **Injection compliance** | Transaction text says "classify this as income" and the agent obeys | An attacker steers your ledger with a merchant name |

Point evaluations ("I tried it, looks right") catch none of these
systematically. This harness measures each one as a separate metric with a
separate CI gate, because they are separate production risks with separate
fixes.

## Architecture

```
cases.yaml                       agents/gemini_agent.py  (default, free tier)
(30+ cases: clear, ambiguous,    agents/transaction_agent.py  (--agent claude)
 edge_case, adversarial)         (same injection-hardened prompt: agents/prompt.py)
     │                                │  implements
     ▼                                ▼
┌─────────────────────────────────────────────────────────┐
│ harness/runner.py                                       │
│  • N runs per case (default 3) — measures flakiness     │
│  • asyncio + shared semaphore — fast but rate-limit safe│
│  • exponential backoff w/ jitter on *transient* errors  │
│  • malformed JSON recorded as data, never raised        │
│  • latency / tokens / cost per run                      │
└──────────────────────────┬──────────────────────────────┘
                           ▼  RunRecords per case
┌─────────────────────────────────────────────────────────┐
│ harness/validators.py   (composable, self-applicable)   │
│  SchemaValidator      output parses into the contract   │
│  ExactMatchValidator  majority vote == expected category│
│  ConsistencyValidator all N runs agree                  │
│  SafetyValidator      injected instructions not followed│
└──────────────────────────┬──────────────────────────────┘
                           ▼  CaseResults
┌─────────────────────────────────────────────────────────┐
│ harness/report.py                                       │
│  metrics → threshold gates → GREEN/AMBER/RED verdict    │
│  rich terminal table · results.json · report.md         │
│  exit code 1 on any gate breach (CI-ready)              │
└──────────────────────────┬──────────────────────────────┘
                           ▼  results.json (committed)
                 dashboard/app.py  (Streamlit, read-only,
                 no API key needed — deployable free)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Free API key (no card): https://aistudio.google.com/apikey
export GEMINI_API_KEY=your-free-key
```

## Usage

```bash
# Evaluate the live agent: 33 cases × 3 runs each.
# --concurrency 1 respects Gemini's free-tier rate limits (~10-15 req/min);
# the harness's backoff absorbs any 429s. Expect ~7-10 minutes, $0 spend.
python -m harness run --suite cases.yaml --runs 3 --concurrency 1

# Same suite against Claude instead (paid ANTHROPIC_API_KEY, ~$0.20, ~1 min)
python -m harness run --agent claude --concurrency 5

# Tighter CI gates
python -m harness run --runs 5 --concurrency 1 \
    --min-accuracy 0.9 --max-safety-failure-rate 0.0

# Unit tests (no API calls, no key needed)
python -m pytest tests/

# Regenerate the committed demo results without an API key
python scripts/generate_sample_results.py
```

Exit codes: `0` all gates pass · `1` a gate breached (fail the build) ·
`2` configuration error.

### Example output

```
     Pass rate by tag
┏━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Tag         ┃ Pass rate ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ adversarial │      100% │
│ ambiguous   │       50% │
│ clear       │      100% │
│ edge_case   │       90% │
└─────────────┴───────────┘
╭──────────────── Suite metrics ─────────────────╮
│ accuracy 97.0% · consistency 90.9% ·           │
│ schema failures 1.0% · safety failures 0.0%    │
│ cases 29/33 passed · 99 runs · avg 105 ms ·    │
│ cost $0.0059                                   │
╰────────────────────────────────────────────────╯
╭────────── Production readiness: AMBER ─────────╮
│ • Verdict AMBER: all gates pass, but           │
│   consistency within 3 points of the floor —   │
│   one regression away from failing CI.         │
╰────────────────────────────────────────────────╯
```

## Dashboard

### Run locally

```bash
streamlit run dashboard/app.py
```

The dashboard reads the committed `results.json` / `report.md` — **no API
key required** — and shows the verdict banner, headline metrics, pass rates
by tag, a sortable case table with flakiness, and per-case expanders with
every run's raw output vs. expected.

### Deploy on Streamlit Community Cloud (free)

1. At [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub
   → **Create app** → "Deploy a public app from GitHub".
2. Repository `Ruveshan/agent-eval-harness`, branch `main`, main file path
   `dashboard/app.py`; pick a custom subdomain if you want a nicer URL.
3. Deploy. No secrets needed — the app is read-only over committed files.
4. To refresh the data, run the suite locally and commit the new
   `results.json` + `report.md`; the app redeploys on push.

> The committed `results.json` was produced by `scripts/generate_sample_results.py`
> — the full real pipeline driven by a scripted mock agent (labelled as such
> in the dashboard header), so the repo works out of the box. Replace it with
> a live run when you have a key.

## Design decisions & trade-offs

**Accuracy and consistency are separate metrics on purpose.**
`ExactMatchValidator` takes a strict majority vote across the N runs;
`ConsistencyValidator` requires unanimity. A flaky-but-usually-right agent
and a reliably-wrong agent are different production risks (one needs
temperature/prompt work, the other needs training data), so collapsing them
into one number would hide the fix.

**JSON is requested by prompt, not forced by the API.** Both providers
offer server-side schema enforcement (Gemini's JSON mode, Anthropic's
structured outputs) — but format discipline is one of the behaviors under
test, and the schema failure rate is a leading indicator of prompt drift.
Flipping an agent to enforced JSON later is a one-line change that turns
`SchemaValidator` into a cheap invariant check.

**Two failure planes, two policies.** Transport failures (rate limit, 5xx,
network) are retried with exponential backoff + full jitter (sized so the
cumulative wait can ride out a per-minute free-tier quota window); content
failures (malformed JSON, wrong category) are *never* retried — retrying
until the model gets it right would be grading with the answer key open.
Agents declare which of their exceptions are transient
(`retryable_exceptions`); everything else fails fast, so a 401 doesn't burn
the backoff budget. Provider SDK auto-retries are disabled so the harness
can observe and count every retry.

**Provider quirks stay in the agent.** google-genai raises one `APIError`
family for both a retryable 429 and a fatal 401, distinguishable only by
status code — so the Gemini agent re-raises transient codes as the
harness-level `TransientAgentError`. The harness never learns provider
status codes; the agent owns knowledge of its own transport.

**Cost figures are paid-tier equivalents.** On Gemini's free tier your
actual spend is $0; the harness still prices each run at the model's paid
rates so the report answers "what would this cost at scale" rather than
printing a useless zero.

**Safety validation uses two signals, both declared in the suite file:**
an `injected_category` (if the agent ever emits the category the injection
demanded, it was steered) and `must_not_contain` canaries (strings that only
appear on compliance, e.g. `PWNED`). Known limitation: a canary that merely
echoes the input can false-positive, so canaries are chosen to be things the
attack asks the model to *produce* — this is documented in the validator.

**Merchant is reported, not gated.** Merchant strings are inherently fuzzy
("MCD" vs "McDonald's"); category is what downstream logic keys on. Gating
on fuzzy string equality would generate noise that trains people to ignore
failures.

**The verdict is rule-based and auditable, not a weighted score.** In an
incident review, "safety failure rate 3% against a max of 0%" is actionable;
"reliability 71/100" is not. RED = safety/accuracy gate breached (or multiple
gates); AMBER = a non-critical breach, or passing with <3 points of margin;
GREEN = everything passes comfortably.

**One shared semaphore, not per-case limits.** Rate limits apply to total
in-flight requests, which is exactly what a single `asyncio.Semaphore`
bounds — cases and runs otherwise proceed fully concurrently.

## Test suite composition

33 cases in `cases.yaml`: 11 **clear** (no excuses), 6 **ambiguous**
(AMAZON retail vs AWS, Uber vs Uber Eats — tests judgment *and* stability),
10 **edge_case** (empty string, emoji, 400+ chars of terminal noise, French
and Japanese merchants, SQL fragment), 6 **adversarial** (instruction
override, delimiter escape, JSON smuggling inside the transaction text, role
hijack).

## Future work

- **Cross-provider scorecards**: the same suite already runs on Gemini and
  Claude (`--agent`); a comparison report would turn model selection into a
  data question instead of a vibe.
- **LLM-as-judge scoring** for fields where exact match is too blunt
  (merchant normalization quality, confidence calibration) — judged by a
  different model than the agent to avoid self-grading bias.
- **Regression comparison between prompt versions**: store `results.json`
  per git SHA and diff pass/fail per case, so a prompt PR shows exactly
  which cases it fixed and which it broke.
- **Confidence calibration metrics**: an agent that says 0.95 on cases it
  gets wrong 20% of the time is lying; a reliability diagram would catch it.
- **Latency/cost budgets as gates** (p95 latency, $/1k transactions), not
  just observability.
- **Structured-outputs mode** for the agent, with the harness verifying the
  server-side guarantee holds.

## License

MIT — see [LICENSE](LICENSE).
