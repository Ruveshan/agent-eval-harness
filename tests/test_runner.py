"""Runner behavior: parsing tolerance, retry semantics, flakiness math.

Tests drive the real async runner with scripted agents via asyncio.run —
no event-loop plugin needed, no network involved.
"""

from __future__ import annotations

import asyncio

from harness.models import Category
from harness.runner import RetryPolicy, Runner, parse_agent_output
from harness.validators import DEFAULT_VALIDATORS

from .conftest import ScriptedAgent, make_case, output_json

FAST_RETRIES = RetryPolicy(max_attempts=3, base_delay_s=0.0, max_delay_s=0.0)


def run_case(agent, case, runs=3):
    runner = Runner(
        agent=agent,
        validators=DEFAULT_VALIDATORS,
        runs_per_case=runs,
        concurrency=2,
        retry_policy=FAST_RETRIES,
    )
    results, _, _ = asyncio.run(runner.run_suite([case]))
    return results[0]


class TestParseAgentOutput:
    def test_plain_json(self):
        parsed, err = parse_agent_output(output_json("transport", "Uber"))
        assert err is None
        assert parsed.category == Category.TRANSPORT
        assert parsed.merchant == "Uber"

    def test_fenced_json_is_tolerated(self):
        parsed, err = parse_agent_output(f"```json\n{output_json('dining')}\n```")
        assert err is None and parsed.category == Category.DINING

    def test_prose_around_json_is_a_failure(self):
        parsed, err = parse_agent_output(f"Sure! Here you go: {output_json('dining')}")
        assert parsed is None and "invalid JSON" in err

    def test_unknown_category_is_a_schema_violation(self):
        parsed, err = parse_agent_output(output_json("shopping"))
        assert parsed is None and "category" in err

    def test_extra_fields_are_rejected(self):
        parsed, err = parse_agent_output(
            '{"category": "other", "merchant": null, "confidence": 0.5, "note": "hi"}'
        )
        assert parsed is None and "note" in err

    def test_out_of_range_confidence_is_rejected(self):
        parsed, err = parse_agent_output(
            '{"category": "other", "merchant": null, "confidence": 1.7}'
        )
        assert parsed is None and "confidence" in err


class TestRetries:
    def test_transient_errors_are_retried_then_succeed(self):
        agent = ScriptedAgent(
            [ConnectionError("boom"), ConnectionError("boom"), output_json("transport")]
        )
        result = run_case(agent, make_case(expected_category="transport"), runs=1)
        run = result.runs[0]
        assert run.parsed is not None and run.retries == 2
        assert agent.calls == 3

    def test_exhausted_retries_record_an_error_not_a_crash(self):
        agent = ScriptedAgent([ConnectionError("down")])
        result = run_case(agent, make_case(expected_category="transport"), runs=1)
        run = result.runs[0]
        assert run.raw_output is None and run.parsed is None
        assert "ConnectionError" in run.error
        assert agent.calls == FAST_RETRIES.max_attempts

    def test_transient_error_with_hint_still_retries_to_success(self):
        from harness.runner import TransientAgentError

        agent = ScriptedAgent(
            [
                TransientAgentError("quota window", retry_after_s=0.01),
                output_json("transport"),
            ]
        )
        agent.retryable_exceptions = (TransientAgentError,)
        result = run_case(agent, make_case(expected_category="transport"), runs=1)
        assert result.runs[0].parsed is not None and result.runs[0].retries == 1

    def test_non_retryable_errors_fail_fast(self):
        # Agent declares only ConnectionError as transient; a ValueError
        # (think: 401 auth) must be recorded immediately with no retries.
        agent = ScriptedAgent([ValueError("invalid api key")])
        agent.retryable_exceptions = (ConnectionError,)
        result = run_case(agent, make_case(), runs=1)
        run = result.runs[0]
        assert run.raw_output is None and "ValueError" in run.error
        assert run.retries == 0
        assert agent.calls == 1

    def test_malformed_json_is_recorded_not_retried(self):
        agent = ScriptedAgent(["this is not json"])
        result = run_case(agent, make_case(), runs=1)
        run = result.runs[0]
        assert run.raw_output == "this is not json"
        assert run.parsed is None and "invalid JSON" in run.error
        assert agent.calls == 1  # content failures must not burn retries


class TestSuiteBehavior:
    def test_flakiness_reflects_disagreement(self):
        agent = ScriptedAgent(
            [output_json("dining"), output_json("dining"), output_json("other")]
        )
        result = run_case(agent, make_case(expected_category="dining"), runs=3)
        assert abs(result.flakiness - (1 / 3)) < 1e-3  # stored rounded to 4 dp
        by_name = {v.validator: v for v in result.validations}
        assert by_name["exact_match"].passed  # majority still right...
        assert not by_name["consistency"].passed  # ...but unstable

    def test_stable_correct_case_passes_everything(self):
        agent = ScriptedAgent([output_json("groceries", "Woolworths")])
        result = run_case(
            agent,
            make_case(expected_category="groceries", expected_merchant="Woolworths"),
            runs=3,
        )
        assert result.passed and result.flakiness == 0.0

    def test_cost_and_usage_are_tracked(self):
        agent = ScriptedAgent([output_json("other")])
        agent.model = "claude-sonnet-4-6"  # priced model
        result = run_case(agent, make_case(), runs=2)
        run = result.runs[0]
        assert run.input_tokens == 100 and run.output_tokens == 30
        # 100 in @ $3/MTok + 30 out @ $15/MTok
        assert abs(run.cost_usd - (100 * 3 + 30 * 15) / 1_000_000) < 1e-9

    def test_results_preserve_suite_order(self):
        agent = ScriptedAgent([output_json("other")])
        cases = [make_case(id=f"case_{i}") for i in range(5)]
        runner = Runner(agent, DEFAULT_VALIDATORS, runs_per_case=1, retry_policy=FAST_RETRIES)
        results, _, _ = asyncio.run(runner.run_suite(cases))
        assert [r.case.id for r in results] == [c.id for c in cases]
