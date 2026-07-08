"""Verdict logic and metric aggregation."""

from __future__ import annotations

from harness.models import CaseResult, Thresholds, ValidationResult
from harness.report import compute_verdict, summarize

from .conftest import make_case, make_run


def make_case_result(passed: bool, validator: str = "exact_match", tags=None) -> CaseResult:
    case = make_case(tags=tags or ["clear"], expected_category="other")
    return CaseResult(
        case=case,
        runs=[make_run("other")],
        validations=[ValidationResult(validator=validator, passed=passed, reason="…")],
        passed=passed,
        flakiness=0.0,
    )


class TestSummarize:
    def test_accuracy_uses_only_applicable_cases(self):
        cases = [
            make_case_result(True),
            make_case_result(False),
            # a case with no exact_match validation at all:
            make_case_result(True, validator="schema"),
        ]
        metrics = summarize(cases)
        assert metrics.accuracy == 0.5

    def test_tag_pass_rates(self):
        cases = [
            make_case_result(True, tags=["clear"]),
            make_case_result(False, tags=["clear"]),
            make_case_result(True, tags=["adversarial"]),
        ]
        metrics = summarize(cases)
        assert metrics.tag_pass_rates == {"adversarial": 1.0, "clear": 0.5}


class TestVerdict:
    def test_green_when_all_gates_pass_with_margin(self):
        cases = [make_case_result(True) for _ in range(10)]
        verdict = compute_verdict(summarize(cases), Thresholds())
        assert verdict.color == "green" and not verdict.breaches

    def test_safety_breach_is_red(self):
        cases = [make_case_result(True) for _ in range(9)]
        cases.append(make_case_result(False, validator="safety", tags=["adversarial"]))
        verdict = compute_verdict(summarize(cases), Thresholds())
        assert verdict.color == "red"
        assert "safety failure rate" in verdict.breaches

    def test_accuracy_breach_is_red(self):
        cases = [make_case_result(True) for _ in range(6)] + [
            make_case_result(False) for _ in range(4)
        ]
        verdict = compute_verdict(summarize(cases), Thresholds())
        assert verdict.color == "red"
        assert "accuracy" in verdict.breaches

    def test_thin_margin_is_amber(self):
        # 87.5% accuracy passes the 85% gate but sits within 3 points of it.
        cases = [make_case_result(True) for _ in range(7)] + [make_case_result(False)]
        verdict = compute_verdict(summarize(cases), Thresholds(min_consistency=0.0))
        assert verdict.color == "amber" and not verdict.breaches
