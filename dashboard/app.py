"""Streamlit dashboard for harness results.

Read-only by design: it renders a committed `results.json` + `report.md`
and needs no API key, so it can be deployed on Streamlit Community Cloud
straight from the repo. Anyone triaging a failed CI run gets the same
view as the person who ran the suite locally.

Run locally:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "results.json"
REPORT_PATH = ROOT / "report.md"

# Single validated hue for the one-series bar chart (light/dark variants
# from the reference palette; status colors are reserved for the verdict
# banner, where Streamlit pairs them with an icon + text).
ACCENT_LIGHT, ACCENT_DARK = "#2a78d6", "#3987e5"


def accent_color() -> str:
    try:  # st.context.theme requires a recent Streamlit; default to light.
        return ACCENT_DARK if st.context.theme.type == "dark" else ACCENT_LIGHT
    except Exception:
        return ACCENT_LIGHT


@st.cache_data
def load_results(path_str: str, mtime: float) -> dict:
    """mtime participates in the cache key so a regenerated results.json
    invalidates the cache without a manual refresh."""
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def render_verdict(verdict: dict) -> None:
    reasoning = "\n".join(f"- {r}" for r in verdict["reasoning"])
    text = f"**Production readiness: {verdict['color'].upper()}**\n{reasoning}"
    if verdict["color"] == "green":
        st.success(text, icon="✅")
    elif verdict["color"] == "amber":
        st.warning(text, icon="⚠️")
    else:
        st.error(text, icon="🛑")


def render_metrics(metrics: dict) -> None:
    cols = st.columns(6)
    cols[0].metric("Accuracy", f"{metrics['accuracy']:.1%}",
                   help="Cases where the strict majority of runs matched the expected category")
    cols[1].metric("Consistency", f"{metrics['consistency']:.1%}",
                   help="Cases where all runs agreed on the category")
    cols[2].metric("Schema failures", f"{metrics['schema_failure_rate']:.1%}",
                   help="Share of runs whose output was not valid JSON matching the contract")
    cols[3].metric("Safety failures", f"{metrics['safety_failure_rate']:.1%}",
                   help="Share of adversarial cases where injected instructions leaked into output")
    cols[4].metric("Total cost", f"${metrics['total_cost_usd']:.4f}",
                   help=f"{metrics['total_input_tokens']:,} input / "
                        f"{metrics['total_output_tokens']:,} output tokens")
    cols[5].metric("Avg latency", f"{metrics['avg_latency_ms']:.0f} ms",
                   help="Mean latency per successful agent call")


def render_tag_chart(metrics: dict, cases: list[dict]) -> None:
    counts: dict[str, int] = {}
    for c in cases:
        for tag in c["case"]["tags"]:
            counts[tag] = counts.get(tag, 0) + 1
    df = pd.DataFrame(
        [
            {"tag": tag, "pass_rate": rate, "cases": counts.get(tag, 0)}
            for tag, rate in metrics["tag_pass_rates"].items()
        ]
    )
    base = alt.Chart(df).encode(
        y=alt.Y("tag:N", sort="-x", title=None, axis=alt.Axis(grid=False)),
        x=alt.X(
            "pass_rate:Q",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format=".0%", title="pass rate"),
        ),
        tooltip=[
            alt.Tooltip("tag:N", title="tag"),
            alt.Tooltip("pass_rate:Q", format=".1%", title="pass rate"),
            alt.Tooltip("cases:Q", title="cases"),
        ],
    )
    bars = base.mark_bar(
        size=22, cornerRadiusTopRight=4, cornerRadiusBottomRight=4,
        color=accent_color(),
    )
    # Direct labels inside the data end: white on the accent passes
    # contrast in both themes, and stays inside the [0,1] domain.
    labels = base.mark_text(align="right", dx=-6, color="#ffffff").encode(
        text=alt.Text("pass_rate:Q", format=".0%")
    )
    st.altair_chart(
        (bars + labels).properties(height=max(48 * len(df), 96)),
        width="stretch",
    )


def failure_summary(case_result: dict) -> str:
    return "; ".join(
        f"[{v['validator']}] {v['reason']}"
        for v in case_result["validations"]
        if not v["passed"]
    )


def render_case_table(cases: list[dict]) -> None:
    df = pd.DataFrame(
        [
            {
                "case": c["case"]["id"],
                "tags": ", ".join(c["case"]["tags"]),
                "passed": c["passed"],
                "flakiness": c["flakiness"],
                "avg latency (ms)": round(
                    sum(r["latency_ms"] for r in c["runs"]) / len(c["runs"]), 1
                ),
                "failure reasons": failure_summary(c) or "—",
            }
            for c in cases
        ]
    )
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "passed": st.column_config.CheckboxColumn("pass"),
            "flakiness": st.column_config.ProgressColumn(
                "flakiness", min_value=0.0, max_value=1.0, format="percent"
            ),
        },
    )


def render_case_details(cases: list[dict]) -> None:
    only_failures = st.toggle("Show failing cases only", value=False)
    for c in cases:
        if only_failures and c["passed"]:
            continue
        icon = "✅" if c["passed"] else "❌"
        with st.expander(f"{icon} {c['case']['id']}  ·  {', '.join(c['case']['tags'])}"):
            st.markdown(f"**Input:** `{c['case']['input'] or '<empty string>'}`")
            expected = {
                "category": c["case"]["expected_category"],
                "merchant": c["case"]["expected_merchant"],
            }
            st.markdown(f"**Expected:** `{json.dumps(expected)}`")
            for v in c["validations"]:
                mark = "✅" if v["passed"] else "❌"
                st.markdown(f"{mark} **{v['validator']}** — {v['reason']}")
            st.markdown("**Runs:**")
            for i, run in enumerate(c["runs"]):
                meta = (
                    f"run {i} · {run['latency_ms']:.0f} ms · "
                    f"{run['input_tokens']}→{run['output_tokens']} tok · "
                    f"${run['cost_usd']:.5f}"
                    + (f" · {run['retries']} retries" if run["retries"] else "")
                )
                st.caption(meta)
                if run["raw_output"] is not None:
                    st.code(run["raw_output"], language="json")
                if run["error"]:
                    st.caption(f"⚠️ {run['error']}")


def main() -> None:
    st.set_page_config(
        page_title="Agent Evaluation Dashboard", page_icon="🧪", layout="wide"
    )
    st.title("🧪 Agent Evaluation Dashboard")

    if not RESULTS_PATH.exists():
        st.error(
            "No `results.json` found in the repo root. Run "
            "`python -m harness run --suite cases.yaml` first, or commit a results file."
        )
        st.stop()

    data = load_results(str(RESULTS_PATH), RESULTS_PATH.stat().st_mtime)
    st.caption(
        f"Agent **{data['agent_name']}** on `{data['model']}` · "
        f"{data['runs_per_case']} runs per case · {data['started_at']} → {data['finished_at']}"
    )

    render_verdict(data["verdict"])
    render_metrics(data["metrics"])

    st.subheader("Pass rate by tag")
    render_tag_chart(data["metrics"], data["cases"])

    st.subheader("All test cases")
    render_case_table(data["cases"])

    st.subheader("Case details")
    render_case_details(data["cases"])

    if REPORT_PATH.exists():
        with st.expander("📄 Full report.md"):
            st.markdown(REPORT_PATH.read_text(encoding="utf-8"))


main()
