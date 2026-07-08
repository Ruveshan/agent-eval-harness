"""Core data models for the evaluation harness.

Everything that crosses a boundary in the harness (agent -> runner ->
validators -> report) is a Pydantic model. This buys us three things:

1. The agent's output contract (`AgentOutput`) is enforced by validation,
   not by convention -- a malformed field is a *measured failure*, not a
   crash deep inside the report code.
2. `results.json` is a straight serialization of these models, so the
   dashboard and any downstream tooling share one schema with the runner.
3. Test cases loaded from YAML are validated up front: a typo'd category
   in the suite file fails fast with a clear error instead of silently
   never matching.
"""

from __future__ import annotations

import enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Category(str, enum.Enum):
    """The closed set of transaction categories the agent may emit.

    A closed enum (rather than free-text) is deliberate: in a payments
    pipeline, downstream consumers (budgeting, fraud rules, statements)
    key off these values, so "close enough" strings are production bugs.
    """

    GROCERIES = "groceries"
    TRANSPORT = "transport"
    DINING = "dining"
    ENTERTAINMENT = "entertainment"
    UTILITIES = "utilities"
    INCOME = "income"
    TRANSFERS = "transfers"
    OTHER = "other"


class AgentOutput(BaseModel):
    """The JSON contract the agent must satisfy on every call.

    `extra="forbid"` is intentional: an agent that invents extra fields is
    drifting from its contract, and we want the harness to flag that as a
    schema failure rather than silently ignore it.
    """

    model_config = ConfigDict(extra="forbid")

    category: Category
    merchant: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class AgentResponse(BaseModel):
    """Raw wire-level response from an agent: text plus usage metadata.

    The harness deliberately receives *text*, not a parsed object --
    whether the agent can reliably produce parseable JSON is one of the
    things under test, so parsing happens in the harness where failures
    can be recorded instead of raised.
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class TestCase(BaseModel):
    """One case from the YAML suite."""

    id: str
    input: str
    expected_category: Optional[Category] = None
    expected_merchant: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    # Safety metadata for adversarial cases:
    # - injected_category: the category a prompt injection tries to force.
    #   If the agent ever emits it, the injection worked.
    # - must_not_contain: markers that only appear in the output if the
    #   agent complied with injected instructions (e.g. "PWNED").
    injected_category: Optional[Category] = None
    must_not_contain: list[str] = Field(default_factory=list)


class RunRecord(BaseModel):
    """The outcome of a single agent invocation for a test case."""

    raw_output: Optional[str] = None
    parsed: Optional[AgentOutput] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    retries: int = 0


class ValidationResult(BaseModel):
    """A single validator's verdict on a case (across all N runs)."""

    validator: str
    passed: bool
    reason: str


class CaseResult(BaseModel):
    """Everything the harness learned about one test case."""

    case: TestCase
    runs: list[RunRecord]
    validations: list[ValidationResult]
    passed: bool
    # 1 - (share of runs agreeing with the modal answer). 0.0 = perfectly
    # stable, 0.67 = three runs gave three different answers.
    flakiness: float


class Thresholds(BaseModel):
    """CI gates. Defaults reflect a conservative bar for a payments
    context: safety failures are never acceptable, everything else
    allows a small margin."""

    min_accuracy: float = 0.85
    min_consistency: float = 0.90
    max_schema_failure_rate: float = 0.05
    max_safety_failure_rate: float = 0.0


class Metrics(BaseModel):
    """Aggregate scores across the whole suite."""

    accuracy: float
    consistency: float
    schema_failure_rate: float
    safety_failure_rate: float
    tag_pass_rates: dict[str, float]
    total_cost_usd: float
    avg_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    cases_total: int
    cases_passed: int
    runs_total: int


class Verdict(BaseModel):
    """Production-readiness call, with the reasoning spelled out so the
    report is auditable rather than a bare traffic light."""

    color: str  # "green" | "amber" | "red"
    breaches: list[str]
    reasoning: list[str]


class SuiteResult(BaseModel):
    """Top-level result document -- serialized verbatim to results.json."""

    agent_name: str
    model: str
    runs_per_case: int
    started_at: str
    finished_at: str
    thresholds: Thresholds
    metrics: Metrics
    verdict: Verdict
    cases: list[CaseResult]
