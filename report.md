# Agent Evaluation Report

- **Agent:** transaction-classifier [MOCK DEMO RUN — regenerate with a real API key] (`claude-sonnet-4-6`)
- **Runs per case:** 3
- **Window:** 2026-07-08T17:18:10+00:00 → 2026-07-08T17:18:12+00:00

## Production readiness: 🟠 AMBER

- accuracy 97.0% meets the >= 85.0% gate
- consistency 90.9% meets the >= 90.0% gate
- schema failure rate 1.0% meets the <= 5.0% gate
- safety failure rate 0.0% meets the <= 0.0% gate
- Verdict AMBER: all gates pass, but consistency within 3 points of the floor — one regression away from failing CI.

## Headline metrics

| Metric | Value | Gate |
|---|---|---|
| Accuracy (majority vote) | 97.0% | ≥ 85% |
| Consistency (all runs agree) | 90.9% | ≥ 90% |
| Schema failure rate | 1.0% | ≤ 5% |
| Safety failure rate | 0.0% | ≤ 0% |
| Cases passed | 29/33 | — |
| Total cost | $0.1876 (44,346 in / 3,636 out tokens) | — |
| Avg latency per call | 106 ms | — |

## Pass rate by tag

| Tag | Pass rate |
|---|---|
| adversarial | 100% |
| ambiguous | 50% |
| clear | 100% |
| edge_case | 90% |

## Worst-performing cases

### `edge_emoji_only` — flakiness 33%

- Input: `✨✨✨`
- **schema**: 1/3 runs failed schema validation (run 1: invalid JSON: Expecting value (pos 0))
- **consistency**: one or more runs produced no parseable output

### `ambiguous_amazon_bare` — flakiness 33%

- Input: `AMAZON`
- **consistency**: runs disagreed across 2 categories: other, utilities

### `ambiguous_apple_bill` — flakiness 33%

- Input: `APPLE.COM/BILL ITUNES.COM`
- **consistency**: runs disagreed across 2 categories: entertainment, other

### `ambiguous_paypal_seller` — flakiness 0%

- Input: `PAYPAL *JOHNSMITH8842`
- **exact_match**: expected 'other' but votes were: transfers=3
