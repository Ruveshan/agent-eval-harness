"""Aggregation, verdict logic, and report rendering.

The verdict is intentionally simple and auditable -- a handful of
threshold comparisons with the reasoning written out -- rather than a
weighted score. In a review or an incident postmortem, "safety failure
rate was 0.08 against a max of 0.0" is actionable; "reliability score
71/100" is not.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .models import CaseResult, Metrics, SuiteResult, Thresholds, Verdict
from .validators import case_failure_summary


def _rate(passed: int, total: int) -> float:
    return round(passed / total, 4) if total else 1.0


def summarize(cases: list[CaseResult]) -> Metrics:
    """Roll per-case results up into suite-level metrics.

    Denominators are 'applicable cases' per validator: accuracy is only
    measured where an expected category exists, safety only on cases that
    carry adversarial checks. This keeps each metric meaningful instead of
    diluting it with not-applicable passes.
    """
    def validator_rate(name: str) -> float:
        applicable = [
            v for c in cases for v in c.validations if v.validator == name
        ]
        if not applicable:
            return 1.0
        return _rate(sum(1 for v in applicable if v.passed), len(applicable))

    all_runs = [r for c in cases for r in c.runs]
    produced = [r for r in all_runs if r.raw_output is not None]
    schema_failures = sum(1 for r in produced if r.parsed is None)

    tags = sorted({t for c in cases for t in c.case.tags})
    tag_pass_rates = {}
    for tag in tags:
        tagged = [c for c in cases if tag in c.case.tags]
        tag_pass_rates[tag] = _rate(sum(1 for c in tagged if c.passed), len(tagged))

    ok_latencies = [r.latency_ms for r in produced]
    return Metrics(
        accuracy=validator_rate("exact_match"),
        consistency=validator_rate("consistency"),
        schema_failure_rate=_rate(schema_failures, len(produced)) if produced else 1.0,
        safety_failure_rate=round(1.0 - validator_rate("safety"), 4),
        tag_pass_rates=tag_pass_rates,
        total_cost_usd=round(sum(r.cost_usd for r in all_runs), 4),
        avg_latency_ms=round(sum(ok_latencies) / len(ok_latencies), 1) if ok_latencies else 0.0,
        total_input_tokens=sum(r.input_tokens for r in all_runs),
        total_output_tokens=sum(r.output_tokens for r in all_runs),
        cases_total=len(cases),
        cases_passed=sum(1 for c in cases if c.passed),
        runs_total=len(all_runs),
    )


def compute_verdict(metrics: Metrics, thresholds: Thresholds) -> Verdict:
    """Traffic-light call with explicit reasoning.

    RED   -- a safety or accuracy gate is breached (the two failure modes
             that directly cost money or trust in a payments product), or
             more than one gate of any kind is breached.
    AMBER -- a single non-critical gate breached, or all gates pass but
             accuracy/consistency sit within 3 points of their floor
             (one bad prompt tweak from failing CI).
    GREEN -- every gate passes with margin.
    """
    t = thresholds
    breaches: list[str] = []
    reasoning: list[str] = []

    checks = [
        ("accuracy", metrics.accuracy, t.min_accuracy, ">="),
        ("consistency", metrics.consistency, t.min_consistency, ">="),
        ("schema failure rate", metrics.schema_failure_rate, t.max_schema_failure_rate, "<="),
        ("safety failure rate", metrics.safety_failure_rate, t.max_safety_failure_rate, "<="),
    ]
    for label, value, bound, op in checks:
        ok = value >= bound if op == ">=" else value <= bound
        if ok:
            reasoning.append(f"{label} {value:.1%} meets the {op} {bound:.1%} gate")
        else:
            breaches.append(label)
            reasoning.append(f"{label} {value:.1%} BREACHES the {op} {bound:.1%} gate")

    critical = {"safety failure rate", "accuracy"}
    if any(b in critical for b in breaches) or len(breaches) > 1:
        color = "red"
        reasoning.append(
            "Verdict RED: do not ship — a critical gate (safety/accuracy) failed "
            "or multiple gates failed."
        )
    elif breaches:
        color = "amber"
        reasoning.append(
            "Verdict AMBER: a non-critical gate failed; fix before relying on this in production."
        )
    else:
        thin = [
            label
            for label, value, bound, op in checks[:2]
            if value - bound < 0.03
        ]
        if thin:
            color = "amber"
            reasoning.append(
                f"Verdict AMBER: all gates pass, but {', '.join(thin)} within 3 points "
                "of the floor — one regression away from failing CI."
            )
        else:
            color = "green"
            reasoning.append("Verdict GREEN: all gates pass with comfortable margin.")
    return Verdict(color=color, breaches=breaches, reasoning=reasoning)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_VERDICT_STYLE = {"green": "bold green", "amber": "bold yellow", "red": "bold red"}


def render_console(result: SuiteResult, console: Console) -> None:
    """Terminal summary: per-case table, tag scores, verdict panel."""
    table = Table(title=f"Suite results — {result.agent_name} ({result.model})")
    table.add_column("Case", style="cyan", no_wrap=True)
    table.add_column("Tags", style="dim")
    table.add_column("Result", justify="center")
    table.add_column("Flaky %", justify="right")
    table.add_column("Failures", max_width=60)
    for c in result.cases:
        table.add_row(
            c.case.id,
            ",".join(c.case.tags),
            "[green]PASS[/green]" if c.passed else "[red]FAIL[/red]",
            f"{c.flakiness:.0%}",
            case_failure_summary(c) or "—",
        )
    console.print(table)

    m = result.metrics
    tag_table = Table(title="Pass rate by tag")
    tag_table.add_column("Tag")
    tag_table.add_column("Pass rate", justify="right")
    for tag, rate in m.tag_pass_rates.items():
        tag_table.add_row(tag, f"{rate:.0%}")
    console.print(tag_table)

    summary = (
        f"accuracy [bold]{m.accuracy:.1%}[/bold] · consistency [bold]{m.consistency:.1%}[/bold] · "
        f"schema failures [bold]{m.schema_failure_rate:.1%}[/bold] · safety failures "
        f"[bold]{m.safety_failure_rate:.1%}[/bold]\n"
        f"cases {m.cases_passed}/{m.cases_total} passed · {m.runs_total} runs · "
        f"avg latency {m.avg_latency_ms:.0f} ms · cost ${m.total_cost_usd:.4f}"
    )
    console.print(Panel(summary, title="Suite metrics"))
    console.print(
        Panel(
            "\n".join(f"• {r}" for r in result.verdict.reasoning),
            title=f"Production readiness: {result.verdict.color.upper()}",
            border_style=_VERDICT_STYLE[result.verdict.color],
        )
    )


def write_results_json(result: SuiteResult, path: Path) -> None:
    path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_report_md(result: SuiteResult, path: Path) -> None:
    """Human-readable report for reviewers who won't open the dashboard."""
    m, v = result.metrics, result.verdict
    icon = {"green": "🟢", "amber": "🟠", "red": "🔴"}[v.color]
    lines = [
        "# Agent Evaluation Report",
        "",
        f"- **Agent:** {result.agent_name} (`{result.model}`)",
        f"- **Runs per case:** {result.runs_per_case}",
        f"- **Window:** {result.started_at} → {result.finished_at}",
        "",
        f"## Production readiness: {icon} {v.color.upper()}",
        "",
        *(f"- {r}" for r in v.reasoning),
        "",
        "## Headline metrics",
        "",
        "| Metric | Value | Gate |",
        "|---|---|---|",
        f"| Accuracy (majority vote) | {m.accuracy:.1%} | ≥ {result.thresholds.min_accuracy:.0%} |",
        f"| Consistency (all runs agree) | {m.consistency:.1%} | ≥ {result.thresholds.min_consistency:.0%} |",
        f"| Schema failure rate | {m.schema_failure_rate:.1%} | ≤ {result.thresholds.max_schema_failure_rate:.0%} |",
        f"| Safety failure rate | {m.safety_failure_rate:.1%} | ≤ {result.thresholds.max_safety_failure_rate:.0%} |",
        f"| Cases passed | {m.cases_passed}/{m.cases_total} | — |",
        f"| Total cost | ${m.total_cost_usd:.4f} ({m.total_input_tokens:,} in / {m.total_output_tokens:,} out tokens) | — |",
        f"| Avg latency per call | {m.avg_latency_ms:.0f} ms | — |",
        "",
        "## Pass rate by tag",
        "",
        "| Tag | Pass rate |",
        "|---|---|",
        *(f"| {tag} | {rate:.0%} |" for tag, rate in m.tag_pass_rates.items()),
        "",
        "## Worst-performing cases",
        "",
    ]
    failing = sorted(
        (c for c in result.cases if not c.passed),
        key=lambda c: (sum(1 for x in c.validations if not x.passed), c.flakiness),
        reverse=True,
    )
    if failing:
        for c in failing:
            lines.append(f"### `{c.case.id}` — flakiness {c.flakiness:.0%}")
            lines.append("")
            lines.append(f"- Input: `{c.case.input[:120]}`")
            for val in c.validations:
                if not val.passed:
                    lines.append(f"- **{val.validator}**: {val.reason}")
            lines.append("")
    else:
        lines.append("All cases passed. 🎉")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
