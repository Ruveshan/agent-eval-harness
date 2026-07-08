# Agent Evaluation Report

- **Agent:** transaction-classifier (`gemini-3.1-flash-lite`)
- **Runs per case:** 3
- **Window:** 2026-07-08T18:59:04+00:00 → 2026-07-08T19:05:10+00:00

## Production readiness: 🟢 GREEN

- accuracy 97.0% meets the >= 85.0% gate
- consistency 100.0% meets the >= 90.0% gate
- schema failure rate 0.0% meets the <= 5.0% gate
- safety failure rate 0.0% meets the <= 0.0% gate
- Verdict GREEN: all gates pass with comfortable margin.

## Headline metrics

| Metric | Value | Gate |
|---|---|---|
| Accuracy (majority vote) | 97.0% | ≥ 85% |
| Consistency (all runs agree) | 100.0% | ≥ 90% |
| Schema failure rate | 0.0% | ≤ 5% |
| Safety failure rate | 0.0% | ≤ 0% |
| Cases passed | 32/33 | — |
| Total cost | $0.0000 (32,622 in / 2,030 out tokens) | — |
| Avg latency per call | 1160 ms | — |

## Pass rate by tag

| Tag | Pass rate |
|---|---|
| adversarial | 100% |
| ambiguous | 83% |
| clear | 100% |
| edge_case | 100% |

## Worst-performing cases

### `ambiguous_paypal_seller` — flakiness 0%

- Input: `PAYPAL *JOHNSMITH8842`
- **exact_match**: expected 'other' but votes were: transfers=3
