"""CLI entry point: `python -m harness run --suite cases.yaml --runs 3`.

Exit code contract (for CI): 0 when every threshold gate passes, 1 when
any gate is breached, 2 on usage/configuration errors. The verdict color
can be AMBER for "passing but thin margin" without failing the build --
only actual gate breaches are non-zero.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

from .models import SuiteResult, TestCase, Thresholds
from .report import (
    compute_verdict,
    render_console,
    summarize,
    write_report_md,
    write_results_json,
)
from .runner import Runner
from .validators import DEFAULT_VALIDATORS


def load_suite(path: Path) -> list[TestCase]:
    with path.open(encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict) or "cases" not in doc:
        raise ValueError(f"{path}: expected a top-level 'cases' list")
    cases = [TestCase.model_validate(raw) for raw in doc["cases"]]
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"{path}: duplicate case id '{case.id}'")
        seen.add(case.id)
    return cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m harness",
        description="Evaluate an LLM agent for accuracy, consistency, and robustness.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a test suite against the agent")
    run.add_argument("--suite", type=Path, default=Path("cases.yaml"),
                     help="YAML suite file (default: cases.yaml)")
    run.add_argument("--runs", type=int, default=3,
                     help="Runs per case for consistency measurement (default: 3)")
    run.add_argument("--concurrency", type=int, default=5,
                     help="Max in-flight API calls (default: 5)")
    run.add_argument("--output-dir", type=Path, default=Path("."),
                     help="Where to write results.json and report.md (default: .)")
    defaults = Thresholds()
    run.add_argument("--min-accuracy", type=float, default=defaults.min_accuracy,
                     help=f"CI gate: minimum majority-vote accuracy (default: {defaults.min_accuracy})")
    run.add_argument("--min-consistency", type=float, default=defaults.min_consistency,
                     help=f"CI gate: minimum all-runs-agree rate (default: {defaults.min_consistency})")
    run.add_argument("--max-schema-failure-rate", type=float,
                     default=defaults.max_schema_failure_rate,
                     help=f"CI gate: max share of runs with invalid output (default: {defaults.max_schema_failure_rate})")
    run.add_argument("--max-safety-failure-rate", type=float,
                     default=defaults.max_safety_failure_rate,
                     help=f"CI gate: max share of adversarial cases compromised (default: {defaults.max_safety_failure_rate})")
    return parser


def _make_agent():
    """Imported lazily so `--help`, tests, and the dashboard never require
    the anthropic package or an API key."""
    from agents.transaction_agent import TransactionAgent

    return TransactionAgent()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()

    try:
        cases = load_suite(args.suite)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        console.print(f"[red]Failed to load suite:[/red] {exc}")
        return 2

    try:
        agent = _make_agent()
    except Exception as exc:  # missing key, missing package, etc.
        console.print(f"[red]Failed to construct agent:[/red] {exc}")
        return 2

    thresholds = Thresholds(
        min_accuracy=args.min_accuracy,
        min_consistency=args.min_consistency,
        max_schema_failure_rate=args.max_schema_failure_rate,
        max_safety_failure_rate=args.max_safety_failure_rate,
    )
    runner = Runner(
        agent=agent,
        validators=DEFAULT_VALIDATORS,
        runs_per_case=args.runs,
        concurrency=args.concurrency,
    )

    console.print(
        f"Running [bold]{len(cases)}[/bold] cases × [bold]{args.runs}[/bold] runs "
        f"against [bold]{agent.name}[/bold] ({agent.model})…"
    )
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating", total=len(cases))
        results, started_at, finished_at = asyncio.run(
            runner.run_suite(cases, on_case_complete=lambda _: progress.advance(task))
        )

    metrics = summarize(results)
    verdict = compute_verdict(metrics, thresholds)
    suite_result = SuiteResult(
        agent_name=agent.name,
        model=agent.model,
        runs_per_case=args.runs,
        started_at=started_at,
        finished_at=finished_at,
        thresholds=thresholds,
        metrics=metrics,
        verdict=verdict,
        cases=results,
    )

    render_console(suite_result, console)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.json"
    report_path = args.output_dir / "report.md"
    write_results_json(suite_result, results_path)
    write_report_md(suite_result, report_path)
    console.print(f"Wrote [bold]{results_path}[/bold] and [bold]{report_path}[/bold]")

    return 1 if verdict.breaches else 0


if __name__ == "__main__":
    sys.exit(main())
