"""Generate a demo results.json/report.md WITHOUT calling the Anthropic API.

Why this exists: the dashboard must be deployable (and reviewable) from a
committed results file, but a fresh checkout has no API key. This script
runs the *entire real pipeline* — runner, retries, validators, metrics,
verdict, report writers — against a scripted agent whose answers imitate a
realistic model: mostly right, flaky on ambiguous merchants, one malformed
response, injections resisted. The agent name in the output is labelled
MOCK so nobody mistakes it for a live evaluation.

Regenerate with real data:  export ANTHROPIC_API_KEY=... && python -m harness run

Usage:  python scripts/generate_sample_results.py
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console

from harness.cli import load_suite
from harness.models import AgentResponse, SuiteResult, Thresholds
from harness.report import (
    compute_verdict,
    render_console,
    summarize,
    write_report_md,
    write_results_json,
)
from harness.runner import Runner
from harness.validators import DEFAULT_VALIDATORS

rng = random.Random(2026)  # deterministic-ish demo output


def answer(category: str, merchant: str | None, confidence: float) -> str:
    return json.dumps(
        {"category": category, "merchant": merchant, "confidence": confidence}
    )


# Per-case scripted responses (cycled across the 3 runs). Anything not
# listed here answers with the suite's expected values — i.e. a good run.
SCRIPTED: dict[str, list[str]] = {
    # Flaky on genuinely ambiguous merchants: majority right, one dissent.
    "ambiguous_amazon_bare": [
        answer("other", "Amazon", 0.52),
        answer("other", "Amazon", 0.55),
        answer("utilities", "Amazon", 0.48),
    ],
    "ambiguous_apple_bill": [
        answer("entertainment", "Apple", 0.61),
        answer("other", "Apple", 0.5),
        answer("entertainment", "Apple", 0.64),
    ],
    # Reliably wrong: plausible but not the labelled category.
    "ambiguous_paypal_seller": [answer("transfers", "PayPal", 0.58)],
    # One malformed response out of three (prose instead of JSON).
    "edge_emoji_only": [
        answer("other", None, 0.2),
        "I could not identify a transaction in this input.",
        answer("other", None, 0.25),
    ],
}


class MockTransactionAgent:
    """Deterministic stand-in for TransactionAgent. Simulates latency so
    the demo dashboard shows plausible timing, and reports token counts in
    the range the real prompt produces, so cost figures are representative."""

    name = "transaction-classifier [MOCK DEMO RUN — regenerate with a real API key]"
    model = "claude-sonnet-4-6"

    def __init__(self, suite_path: Path) -> None:
        self._queues: dict[str, list[str]] = {}
        for case in load_suite(suite_path):
            script = SCRIPTED.get(case.id)
            if script is None:
                expected = case.expected_category.value if case.expected_category else "other"
                script = [
                    answer(expected, case.expected_merchant, round(rng.uniform(0.82, 0.98), 2))
                ]
            self._queues[case.input] = list(script)

    async def run(self, input_text: str) -> AgentResponse:
        await asyncio.sleep(rng.uniform(0.05, 0.15))  # simulated network time
        queue = self._queues[input_text]
        text = queue.pop(0)
        queue.append(text)  # cycle
        return AgentResponse(
            text=text,
            input_tokens=rng.randint(430, 470),
            output_tokens=rng.randint(28, 44),
        )


def main() -> None:
    suite_path = ROOT / "cases.yaml"
    cases = load_suite(suite_path)
    agent = MockTransactionAgent(suite_path)
    runner = Runner(agent, DEFAULT_VALIDATORS, runs_per_case=3, concurrency=8)
    results, started_at, finished_at = asyncio.run(runner.run_suite(cases))

    metrics = summarize(results)
    thresholds = Thresholds()
    suite_result = SuiteResult(
        agent_name=agent.name,
        model=agent.model,
        runs_per_case=3,
        started_at=started_at,
        finished_at=finished_at,
        thresholds=thresholds,
        metrics=metrics,
        verdict=compute_verdict(metrics, thresholds),
        cases=results,
    )
    console = Console()
    render_console(suite_result, console)
    write_results_json(suite_result, ROOT / "results.json")
    write_report_md(suite_result, ROOT / "report.md")
    console.print("Wrote results.json and report.md (demo data)")


if __name__ == "__main__":
    main()
